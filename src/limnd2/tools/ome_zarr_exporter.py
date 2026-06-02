from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import webbrowser
import importlib.util
from contextlib import suppress
from pathlib import Path
from shutil import which
from time import perf_counter, time
from typing import Callable
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import limnd2
from limnd2.export_ome_zarr import ensure_ome_zarr_dependencies


DEFAULT_S3_PREFIX = "s3://lim-public-af010c85-0d3e-4156-9378-5adc1bbef7b3/OmeZarrS3"
DEFAULT_CLOUDFRONT_BASE = "https://d18aphkm7l30ub.cloudfront.net"
DEFAULT_CHUNKS = (1, 1, 1, 512, 512)
DEFAULT_LOCAL_DIR = Path.home() / "Desktop"
COMPANY_INDEX_SCRIPT = Path(r"D:\brainpi_files\bignd2s\list_s3_ome_zarr_validator_links.py")
COMPANY_INDEX_URL = f"{DEFAULT_CLOUDFRONT_BASE}/OmeZarrS3/index.html"


def _aws_cli() -> str:
    aws = which("aws")
    if aws:
        return aws
    fallback = r"C:\Program Files\Amazon\AWSCLIV2\aws.exe"
    if Path(fallback).exists():
        return fallback
    raise RuntimeError("AWS CLI not found on PATH and fallback path does not exist.")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    prefix = "s3://"
    if not uri.startswith(prefix):
        raise ValueError(f"Not an S3 URI: {uri}")
    remainder = uri[len(prefix) :]
    bucket, _, key = remainder.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key.rstrip("/")


def _safe_segment(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _parse_int_tuple(value: str, expected_len: int, *, allow_empty: bool) -> tuple[int, ...] | None:
    cleaned = value.strip()
    if not cleaned:
        if allow_empty:
            return None
        raise ValueError("Value cannot be empty.")
    parts = [part.strip() for part in cleaned.split(",")]
    if len(parts) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated integers.")
    try:
        result = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("All dimensions must be integers.") from exc
    if any(part <= 0 for part in result):
        raise ValueError("All dimensions must be positive integers.")
    return result


def _cloudfront_source_url(dest_uri: str, cloudfront_base: str) -> str:
    bucket_prefix = "s3://lim-public-af010c85-0d3e-4156-9378-5adc1bbef7b3/"
    if not dest_uri.startswith(bucket_prefix):
        raise ValueError(
            "CloudFront mapping is only configured for the company bucket prefix."
        )
    suffix = dest_uri[len(bucket_prefix) :]
    return f"{cloudfront_base.rstrip('/')}/{suffix}/"


def _validator_url(source_url: str) -> str:
    return (
        "https://ome.github.io/ome-ngff-validator/?source="
        f"{quote(source_url, safe=':/?=&')}"
    )


def _napari_command() -> list[str] | None:
    napari_exe = which("napari")
    if napari_exe:
        return [napari_exe]
    if importlib.util.find_spec("napari") is not None:
        return [sys.executable, "-m", "napari"]
    return None


def _launch_napari(target: str | Path) -> None:
    command = _napari_command()
    if command is None:
        raise RuntimeError("Napari is not installed or not available on PATH.")
    subprocess.Popen([*command, str(target)])


def _discover_container_child_source_url(dest_uri: str, cloudfront_base: str) -> str | None:
    aws = _aws_cli()
    bucket, prefix = _parse_s3_uri(dest_uri)
    payload = subprocess.check_output(
        [
            aws,
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            f"{prefix}/",
            "--delimiter",
            "/",
            "--output",
            "json",
        ],
        text=True,
    )
    data = json.loads(payload)
    children: list[str] = []
    for item in data.get("CommonPrefixes", []):
        child_prefix = str(item.get("Prefix", ""))
        if not child_prefix.startswith(f"{prefix}/"):
            continue
        child = child_prefix[len(prefix) + 1 :].strip("/")
        if not child or child == "OME":
            continue
        children.append(child)
    if not children:
        return None
    children.sort(key=lambda item: (not item.isdigit(), int(item) if item.isdigit() else item.lower()))
    root_url = _cloudfront_source_url(dest_uri, cloudfront_base)
    return f"{root_url}{children[0]}/"


def _best_viewer_source_url(dest_uri: str, cloudfront_base: str) -> str:
    root_url = _cloudfront_source_url(dest_uri, cloudfront_base)
    try:
        with urlopen(f"{root_url}zarr.json") as response:
            root_meta = json.load(response)
    except Exception:
        return root_url

    ome = root_meta.get("attributes", {}).get("ome", {})
    if ome.get("multiscales"):
        return root_url

    if ome.get("bioformats2raw.layout") == 3:
        child_url = _discover_container_child_source_url(dest_uri, cloudfront_base)
        if child_url:
            return child_url

    return root_url


def _fix_s3_json_content_types(dest_uri: str, log: Callable[[str], None]) -> None:
    aws = _aws_cli()
    bucket, prefix = _parse_s3_uri(dest_uri)
    token: str | None = None
    json_keys: list[str] = []

    while True:
        args = [
            aws,
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
            "--output",
            "json",
        ]
        if token:
            args += ["--continuation-token", token]
        payload = subprocess.check_output(args, text=True)
        data = json.loads(payload)
        for item in data.get("Contents", []):
            key = str(item["Key"])
            if key.endswith(".json"):
                json_keys.append(key)
        if not data.get("IsTruncated"):
            break
        token = data.get("NextContinuationToken")
        if not token:
            break

    if not json_keys:
        return

    log(f"Rewriting content type for {len(json_keys)} JSON object(s) on S3")
    for key in json_keys:
        uri = f"s3://{bucket}/{key}"
        subprocess.run(
            [
                aws,
                "s3",
                "cp",
                uri,
                uri,
                "--metadata-directive",
                "REPLACE",
                "--content-type",
                "application/json",
                "--only-show-errors",
            ],
            check=True,
        )


def _s3_write_check(dest_prefix: str) -> None:
    aws = _aws_cli()
    bucket, key_prefix = _parse_s3_uri(dest_prefix.rstrip("/"))
    temp_path = Path(tempfile.gettempdir()) / "limnd2_s3_write_check.txt"
    temp_path.write_text("ok\n", encoding="utf-8")
    test_key = f"{key_prefix}/__limnd2_write_check__.txt"
    try:
        subprocess.run(
            [
                aws,
                "s3api",
                "put-object",
                "--bucket",
                bucket,
                "--key",
                test_key,
                "--body",
                str(temp_path),
                "--content-type",
                "text/plain",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                aws,
                "s3api",
                "delete-object",
                "--bucket",
                bucket,
                "--key",
                test_key,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        temp_path.unlink(missing_ok=True)


class OmeZarrExporterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("limnd2 OME-Zarr Exporter")
        self.root.geometry("980x780")

        self.selected_files: list[Path] = []
        self.export_running = False
        self.dependencies_ready = True
        self.napari_command = _napari_command()

        self.mode_var = tk.StringVar(value="local")
        self.local_dir_var = tk.StringVar(value=str(DEFAULT_LOCAL_DIR))
        self.s3_prefix_var = tk.StringVar(value="")
        self.cloudfront_var = tk.StringVar(value=DEFAULT_CLOUDFRONT_BASE)
        self.include_binaries_var = tk.BooleanVar(value=True)
        self.use_dask_var = tk.BooleanVar(value=True)
        self.overwrite_var = tk.BooleanVar(value=True)
        self.open_after_var = tk.BooleanVar(value=True)
        self.open_napari_after_var = tk.BooleanVar(value=False)
        self.update_index_var = tk.BooleanVar(value=False)
        self.chunks_var = tk.StringVar(value="1,1,1,512,512")
        self.shard_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Pick one or more ND2 files to start.")

        self._build_ui()
        self.s3_prefix_var.trace_add("write", lambda *_: self._update_company_index_state())
        self._update_mode_state()
        self.root.after(0, self._check_dependencies_on_startup)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        files_frame = ttk.LabelFrame(outer, text="Input ND2 Files", padding=10)
        files_frame.pack(fill=tk.X)

        files_button_row = ttk.Frame(files_frame)
        files_button_row.pack(fill=tk.X)
        ttk.Button(files_button_row, text="Add Files...", command=self._add_files).pack(side=tk.LEFT)
        ttk.Button(files_button_row, text="Clear", command=self._clear_files).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(files_button_row, textvariable=self.status_var).pack(side=tk.RIGHT)

        self.files_list = tk.Listbox(files_frame, height=8)
        self.files_list.pack(fill=tk.X, pady=(10, 0))

        mode_frame = ttk.LabelFrame(outer, text="Output", padding=10)
        mode_frame.pack(fill=tk.X, pady=(12, 0))

        mode_row = ttk.Frame(mode_frame)
        mode_row.pack(fill=tk.X)
        ttk.Radiobutton(mode_row, text="Local Folder", variable=self.mode_var, value="local", command=self._update_mode_state).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="Generic S3", variable=self.mode_var, value="s3", command=self._update_mode_state).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Radiobutton(mode_row, text="LIM S3", variable=self.mode_var, value="lim_s3", command=self._update_mode_state).pack(side=tk.LEFT, padx=(12, 0))

        self.local_row = ttk.Frame(mode_frame)
        self.local_row.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(self.local_row, text="Local folder:", width=18).pack(side=tk.LEFT)
        self.local_entry = ttk.Entry(self.local_row, textvariable=self.local_dir_var)
        self.local_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.local_button = ttk.Button(self.local_row, text="Browse...", command=self._pick_local_dir)
        self.local_button.pack(side=tk.LEFT, padx=(8, 0))

        self.s3_row = ttk.Frame(mode_frame)
        self.s3_row.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(self.s3_row, text="Generic S3 prefix:", width=18).pack(side=tk.LEFT)
        self.s3_entry = ttk.Entry(self.s3_row, textvariable=self.s3_prefix_var)
        self.s3_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.s3_check_button = ttk.Button(self.s3_row, text="Check write", command=self._check_s3_write_async)
        self.s3_check_button.pack(side=tk.LEFT, padx=(8, 0))

        self.s3_hint = ttk.Label(
            mode_frame,
            text="Example: s3://my-bucket/path/to/exports",
            foreground="#5a6781",
        )
        self.s3_hint.pack(fill=tk.X, pady=(6, 0))

        self.lim_row = ttk.Frame(mode_frame)
        self.lim_row.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(self.lim_row, text="LIM S3 actions:", width=18).pack(side=tk.LEFT)
        self.lim_check_button = ttk.Button(self.lim_row, text="Check write", command=self._check_s3_write_async)
        self.lim_check_button.pack(side=tk.LEFT)
        self.index_button = ttk.Button(self.lim_row, text="Open index", command=self._open_company_index)
        self.index_button.pack(side=tk.LEFT, padx=(8, 0))
        self.index_refresh_button = ttk.Button(
            self.lim_row,
            text="Update index now",
            command=self._refresh_company_index_async,
        )
        self.index_refresh_button.pack(side=tk.LEFT, padx=(8, 0))
        self.index_check = ttk.Checkbutton(
            self.lim_row,
            text="Update index after export",
            variable=self.update_index_var,
        )
        self.index_check.pack(side=tk.LEFT, padx=(8, 0))
        self.lim_hint = ttk.Label(
            mode_frame,
            text="Uses the built-in LIM S3 bucket and CloudFront viewer automatically.",
            foreground="#5a6781",
        )
        self.lim_hint.pack(fill=tk.X, pady=(6, 0))

        options_frame = ttk.LabelFrame(outer, text="Options", padding=10)
        options_frame.pack(fill=tk.X, pady=(12, 0))

        options_top = ttk.Frame(options_frame)
        options_top.pack(fill=tk.X)
        ttk.Checkbutton(options_top, text="Include binaries / labels", variable=self.include_binaries_var).pack(side=tk.LEFT)
        ttk.Checkbutton(options_top, text="Use dask", variable=self.use_dask_var).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Checkbutton(options_top, text="Overwrite existing output", variable=self.overwrite_var).pack(side=tk.LEFT, padx=(12, 0))

        self.options_mid = ttk.Frame(options_frame)
        self.options_mid.pack(fill=tk.X, pady=(10, 0))
        self.open_after_check = ttk.Checkbutton(
            self.options_mid,
            text="Open after export",
            variable=self.open_after_var,
        )
        self.open_napari_check = ttk.Checkbutton(
            self.options_mid,
            text="Open in Napari after export",
            variable=self.open_napari_after_var,
        )

        dims_row = ttk.Frame(options_frame)
        dims_row.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(dims_row, text="Chunks (T,C,Z,Y,X):", width=18).pack(side=tk.LEFT)
        ttk.Entry(dims_row, textvariable=self.chunks_var, width=22).pack(side=tk.LEFT)
        ttk.Label(dims_row, text="Shard shape:", width=12).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(dims_row, textvariable=self.shard_var, width=22).pack(side=tk.LEFT)

        action_row = ttk.Frame(outer)
        action_row.pack(fill=tk.X, pady=(12, 0))
        self.start_button = ttk.Button(action_row, text="Start Export", command=self._start_export)
        self.start_button.pack(side=tk.LEFT)

        log_frame = ttk.LabelFrame(outer, text="Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=20, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _update_mode_state(self) -> None:
        local_enabled = self.mode_var.get() == "local"
        generic_s3_enabled = self.mode_var.get() == "s3"
        lim_s3_enabled = self.mode_var.get() == "lim_s3"
        open_after_enabled = local_enabled or lim_s3_enabled
        napari_open_enabled = self.napari_command is not None and (local_enabled or lim_s3_enabled)

        for widget in (self.local_entry, self.local_button):
            widget.configure(state=tk.NORMAL if local_enabled else tk.DISABLED)
        for widget in (self.s3_entry, self.s3_check_button):
            widget.configure(state=tk.NORMAL if generic_s3_enabled else tk.DISABLED)
        for widget in (
            self.index_button,
            self.index_refresh_button,
            self.index_check,
            self.lim_check_button,
        ):
            widget.configure(state=tk.NORMAL if lim_s3_enabled else tk.DISABLED)

        if local_enabled:
            self.local_row.pack(fill=tk.X, pady=(10, 0))
        else:
            self.local_row.pack_forget()

        if generic_s3_enabled:
            self.s3_row.pack(fill=tk.X, pady=(10, 0))
            self.s3_hint.pack(fill=tk.X, pady=(6, 0))
        else:
            self.s3_row.pack_forget()
            self.s3_hint.pack_forget()

        if lim_s3_enabled:
            self.lim_row.pack(fill=tk.X, pady=(10, 0))
            self.lim_hint.pack(fill=tk.X, pady=(6, 0))
        else:
            self.lim_row.pack_forget()
            self.lim_hint.pack_forget()

        if open_after_enabled:
            self.open_after_check.pack(side=tk.LEFT)
            self.open_after_check.configure(state=tk.NORMAL)
        else:
            self.open_after_check.pack_forget()
            self.open_after_var.set(False)

        if napari_open_enabled:
            self.open_napari_check.pack(side=tk.LEFT, padx=(12, 0))
            self.open_napari_check.configure(state=tk.NORMAL)
        else:
            self.open_napari_check.pack_forget()
            self.open_napari_after_var.set(False)

        self._update_company_index_state()
        if self.napari_command is None:
            self.open_napari_after_var.set(False)

    def _is_company_s3_target(self) -> bool:
        return self.mode_var.get() == "lim_s3"

    def _update_company_index_state(self) -> None:
        enabled = self._is_company_s3_target()
        self.index_button.configure(state=tk.NORMAL if enabled else tk.DISABLED)
        self.index_refresh_button.configure(state=tk.NORMAL if enabled else tk.DISABLED)
        self.index_check.configure(state=tk.NORMAL if enabled else tk.DISABLED)
        if not enabled:
            self.update_index_var.set(False)

    def _set_running(self, running: bool) -> None:
        self.export_running = running
        state = tk.DISABLED if running else tk.NORMAL
        self.start_button.configure(state=state)
        self.s3_check_button.configure(state=state if self.mode_var.get() == "s3" else tk.DISABLED)
        self.lim_check_button.configure(state=state if self.mode_var.get() == "lim_s3" else tk.DISABLED)
        if running:
            self.index_button.configure(state=tk.DISABLED)
            self.index_refresh_button.configure(state=tk.DISABLED)
            self.index_check.configure(state=tk.DISABLED)
        else:
            self._update_company_index_state()
        if not self.dependencies_ready:
            self.start_button.configure(state=tk.DISABLED)
            self.s3_check_button.configure(state=tk.DISABLED)
            self.lim_check_button.configure(state=tk.DISABLED)
            self.index_refresh_button.configure(state=tk.DISABLED)

    def _log(self, message: str) -> None:
        def append() -> None:
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)

        self.root.after(0, append)

    def _set_status(self, message: str) -> None:
        self.root.after(0, lambda: self.status_var.set(message))

    def _check_dependencies_on_startup(self) -> None:
        try:
            ensure_ome_zarr_dependencies(require_s3=True, require_dask=True)
        except Exception as exc:
            self.dependencies_ready = False
            self._log(f"[FAIL] {exc}")
            self._set_status("Missing OME-Zarr dependencies.")
            self._set_running(False)
            messagebox.showerror(
                "Missing OME-Zarr dependencies",
                str(exc),
            )
        else:
            self.dependencies_ready = True
            self._log("OME-Zarr dependencies look available.")
            if self.napari_command is not None:
                self._log(f"Napari available via: {' '.join(self.napari_command)}")
            else:
                self._log("Napari not detected; Napari open option disabled.")

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select ND2 files",
            filetypes=[("ND2 files", "*.nd2"), ("All files", "*.*")],
        )
        if not paths:
            return
        seen = {str(path) for path in self.selected_files}
        for raw in paths:
            if raw not in seen:
                self.selected_files.append(Path(raw))
        self.selected_files.sort(key=lambda path: str(path).lower())
        self._refresh_file_list()

    def _clear_files(self) -> None:
        self.selected_files.clear()
        self._refresh_file_list()

    def _refresh_file_list(self) -> None:
        self.files_list.delete(0, tk.END)
        for path in self.selected_files:
            self.files_list.insert(tk.END, str(path))
        count = len(self.selected_files)
        self.status_var.set(f"{count} file(s) selected" if count else "Pick one or more ND2 files to start.")

    def _pick_local_dir(self) -> None:
        directory = filedialog.askdirectory(title="Select output folder")
        if directory:
            self.local_dir_var.set(directory)

    def _open_company_index(self) -> None:
        if not self._is_company_s3_target():
            messagebox.showinfo(
                "Index unavailable",
                "The company index is only available for the default company S3 prefix.",
            )
            return
        webbrowser.open(f"{COMPANY_INDEX_URL}?cb={int(time())}")

    def _refresh_company_index(self) -> None:
        if not COMPANY_INDEX_SCRIPT.exists():
            raise FileNotFoundError(f"Index script not found: {COMPANY_INDEX_SCRIPT}")
        subprocess.run([sys.executable, str(COMPANY_INDEX_SCRIPT)], check=True)

    def _refresh_company_index_async(self) -> None:
        if self.export_running:
            return
        if not self._is_company_s3_target():
            messagebox.showinfo(
                "Index unavailable",
                "The company index is only available for the default company S3 prefix.",
            )
            return
        if not self.dependencies_ready:
            messagebox.showerror(
                "Missing OME-Zarr dependencies",
                'Install `limnd2[ome-zarr]` before using this exporter.',
            )
            return

        self._set_running(True)
        self._set_status("Refreshing company index...")
        self._log("Refreshing company index.html...")

        def worker() -> None:
            try:
                self._refresh_company_index()
            except Exception as exc:
                self._log(f"Company index refresh failed: {exc}")
                self.root.after(
                    0,
                    lambda: messagebox.showerror("Index refresh failed", str(exc)),
                )
            else:
                self._log("Company index refresh finished.")
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Index refresh",
                        "Company index.html was refreshed successfully.",
                    ),
                )
            finally:
                self._set_status("Ready.")
                self.root.after(0, lambda: self._set_running(False))

        threading.Thread(target=worker, daemon=True).start()

    def _check_s3_write_async(self) -> None:
        if self.export_running:
            return
        if not self.dependencies_ready:
            messagebox.showerror(
                "Missing OME-Zarr dependencies",
                'Install `limnd2[ome-zarr]` before using this exporter.',
            )
            return
        if self.mode_var.get() == "lim_s3":
            prefix = DEFAULT_S3_PREFIX
        else:
            prefix = self.s3_prefix_var.get().strip()
            if not prefix:
                messagebox.showerror("Missing S3 prefix", "Enter an S3 prefix first.")
                return

        self._set_running(True)
        self._set_status("Checking S3 write access...")

        def worker() -> None:
            try:
                _s3_write_check(prefix)
            except Exception as exc:
                self._log(f"S3 write check failed: {exc}")
                self.root.after(0, lambda: messagebox.showerror("S3 write check failed", str(exc)))
            else:
                self._log("S3 write check succeeded.")
                self.root.after(0, lambda: messagebox.showinfo("S3 write check", "Write/delete test succeeded."))
            finally:
                self._set_status("Ready.")
                self.root.after(0, lambda: self._set_running(False))

        threading.Thread(target=worker, daemon=True).start()

    def _collect_settings(self) -> dict[str, object]:
        if not self.selected_files:
            raise ValueError("Select at least one ND2 file.")

        chunks = _parse_int_tuple(self.chunks_var.get(), 5, allow_empty=False)
        shard_shape = _parse_int_tuple(self.shard_var.get(), 5, allow_empty=True)

        mode = self.mode_var.get()
        if mode == "local":
            local_dir = Path(self.local_dir_var.get().strip())
            if not local_dir.exists():
                raise ValueError(f"Local output folder does not exist: {local_dir}")
            s3_prefix = ""
            cloudfront_base = ""
        elif mode == "s3":
            s3_prefix = self.s3_prefix_var.get().strip().rstrip("/")
            if not s3_prefix.startswith("s3://"):
                raise ValueError("S3 output must start with s3://")
            _parse_s3_uri(s3_prefix)
            cloudfront_base = ""
        else:
            s3_prefix = DEFAULT_S3_PREFIX
            _parse_s3_uri(s3_prefix)
            cloudfront_base = self.cloudfront_var.get().strip().rstrip("/")
            if not cloudfront_base:
                raise ValueError("Enter a CloudFront base URL for the LIM S3 mode.")

        return {
            "mode": mode,
            "local_dir": Path(self.local_dir_var.get().strip()),
            "s3_prefix": s3_prefix,
            "cloudfront_base": cloudfront_base,
            "include_binaries": self.include_binaries_var.get(),
            "use_dask": self.use_dask_var.get(),
            "overwrite": self.overwrite_var.get(),
            "open_after": self.open_after_var.get(),
            "open_napari_after": self.open_napari_after_var.get() and self.napari_command is not None,
            "update_index": self.update_index_var.get() and self._is_company_s3_target(),
            "chunks": chunks,
            "shard_shape": shard_shape,
        }

    def _destinations_for_files(self, files: list[Path], settings: dict[str, object]) -> list[str | Path]:
        seen: dict[str, int] = {}
        mode = str(settings["mode"])
        local_dir = Path(settings["local_dir"])
        s3_prefix = str(settings["s3_prefix"])
        destinations: list[str | Path] = []

        for source_path in files:
            safe_base = _safe_segment(source_path.stem)
            count = seen.get(safe_base, 0)
            seen[safe_base] = count + 1
            suffix = f"_{count + 1}" if count else ""
            name = f"{safe_base}{suffix}.ome.zarr"
            if mode == "local":
                destinations.append(local_dir / name)
            else:
                destinations.append(f"{s3_prefix}/{name}")
        return destinations

    def _progress_logger(self, label: str) -> Callable[[int, int, str | Path | None, str], None]:
        state = {"last_bucket": -1, "last_phase": ""}

        def callback(
            current: int, total: int, file: str | Path | None, message: str
        ) -> None:
            if total <= 0:
                return
            percent = (current * 100.0) / total
            bucket = min(100, int(percent // 10) * 10)
            if current == total:
                bucket = 100
            if bucket == state["last_bucket"] and message == state["last_phase"] and current != total:
                return
            state["last_bucket"] = bucket
            state["last_phase"] = message
            self._log(f"[{label}] {percent:5.1f}% ({current}/{total}) file={file} message={message}")

        return callback

    def _start_export(self) -> None:
        if self.export_running:
            return
        if not self.dependencies_ready:
            messagebox.showerror(
                "Missing OME-Zarr dependencies",
                'Install `limnd2[ome-zarr]` before exporting OME-Zarr.',
            )
            return
        try:
            settings = self._collect_settings()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        files = list(self.selected_files)
        destinations = self._destinations_for_files(files, settings)
        self._set_running(True)
        self._set_status("Preparing export...")

        def worker() -> None:
            try:
                self._log("Checking OME-Zarr dependencies...")
                ensure_ome_zarr_dependencies(
                    require_s3=settings["mode"] in {"s3", "lim_s3"},
                    require_dask=True,
                )
                self._log("OME-Zarr dependencies look available.")

                if settings["mode"] in {"s3", "lim_s3"}:
                    self._log("Checking S3 write access...")
                    _s3_write_check(str(settings["s3_prefix"]))
                    self._log("S3 write check succeeded.")

                for index, (source_path, dest) in enumerate(zip(files, destinations), start=1):
                    label = source_path.name
                    self._set_status(f"Exporting {index}/{len(files)}: {label}")
                    self._log("")
                    self._log(f"[{index}/{len(files)}] Exporting {source_path}")
                    self._log(f"[{label}] destination={dest}")

                    start = perf_counter()
                    with limnd2.Nd2Reader(source_path) as reader:
                        nt, nm, nz, ny, nx, nc = reader.imageDataShape
                        dtype = np.dtype(reader.imageAttributes.dtype)
                        approx_gib = (nt * nm * nz * ny * nx * nc * dtype.itemsize) / (1024**3)
                        binary_count = len(list(reader.binaryRasterMetadata))
                        self._log(
                            f"[{label}] shape={(nt, nm, nz, ny, nx, nc)} dtype={dtype} "
                            f"approx={approx_gib:.2f} GiB binaries={binary_count}"
                        )
                        result = reader.to_ome_zarr(
                            dest,
                            overwrite=bool(settings["overwrite"]),
                            use_dask=bool(settings["use_dask"]),
                            chunks=tuple(settings["chunks"]),
                            shard_shape=settings["shard_shape"],
                            include_binaries=bool(settings["include_binaries"]),
                            progress_callback=self._progress_logger(label),
                        )

                    elapsed = perf_counter() - start
                    self._log(f"[{label}] export finished in {elapsed:.2f}s")

                    self._log(f"[{label}] exported to {result}")

                    if bool(settings["open_after"]):
                        if isinstance(dest, Path):
                            os.startfile(str(dest))
                        elif settings["mode"] == "lim_s3":
                            source_url = _best_viewer_source_url(dest, str(settings["cloudfront_base"]))
                            webbrowser.open(_validator_url(source_url))
                        else:
                            self._log(f"[{label}] auto-open skipped for generic S3 target")

                    if bool(settings["open_napari_after"]):
                        if isinstance(dest, Path):
                            _launch_napari(dest)
                        elif settings["mode"] == "lim_s3":
                            source_url = _best_viewer_source_url(
                                dest, str(settings["cloudfront_base"])
                            )
                            _launch_napari(source_url)
                        else:
                            self._log(f"[{label}] Napari auto-open skipped for generic S3 target")

                if bool(settings["update_index"]):
                    self._set_status("Refreshing company index...")
                    self._log("Refreshing company index.html...")
                    self._refresh_company_index()
                    self._log("Company index refresh finished.")

                self._set_status("Export finished.")
                self.root.after(0, lambda: messagebox.showinfo("Export complete", "All exports finished."))
            except Exception as exc:
                self._log(f"[FAIL] {exc}")
                self._set_status("Export failed.")
                self.root.after(0, lambda: messagebox.showerror("Export failed", str(exc)))
            finally:
                self.root.after(0, lambda: self._set_running(False))

        threading.Thread(target=worker, daemon=True).start()


def ome_zarr_exporter_gui() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    with suppress(Exception):
        style.theme_use("vista")
    OmeZarrExporterApp(root)
    root.mainloop()


def main() -> None:
    ome_zarr_exporter_gui()
