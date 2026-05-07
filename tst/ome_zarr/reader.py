import zarr
from pathlib import Path

def inspect_zarr_v3(parent_dir):
    root_path = Path(parent_dir)

    if not root_path.is_dir():
        print(f"Error: {parent_dir} is not a valid directory.")
        return

    # 1. & 2. Find all .ome.zarr folders
    zarr_folders = [f for f in root_path.iterdir() if f.is_dir() and f.name.endswith(".ome.zarr")]

    for z_path in zarr_folders:
        print(f"\n{'='*80}")
        print(f" FILE: {z_path.name}")
        print(f"{'='*80}")

        try:
            # 3. Open the root group
            group = zarr.open(str(z_path), mode='r')

            # Check for your custom 'limnd2' metadata
            attrs = group.attrs.asdict()
            if 'limnd2' in attrs:
                l_meta = attrs['limnd2']
                print(f" Metadata: Axes={l_meta.get('axes')} | Positions={len(l_meta.get('positions', []))}")

            # 4. Recursive search using Zarr 3 'members()'
            def find_arrays(obj, current_path=""):
                # Check if this is an Array (has shape) or a Group (has members/keys)
                if isinstance(obj, zarr.Array):
                    print(f"\n   [Array: /{current_path}]")
                    print(f"    - Shape:    {obj.shape}")
                    print(f"    - Chunks:   {obj.chunks}")

                    # 5. Sharding Check (Zarr 3 specific)
                    if hasattr(obj, 'metadata'):
                        for codec in obj.metadata.codecs:
                            if "sharding" in str(codec).lower():
                                # Pulling the inner chunk/shard shape
                                inner = getattr(codec, 'chunk_shape', 'Unknown')
                                print(f"    - Sharding: ENABLED (Inner Chunk: {inner})")
                                break
                        else:
                            print(f"    - Sharding: DISABLED")

                elif isinstance(obj, zarr.Group):
                    # In Zarr 3, use .members().items() instead of .items()
                    # Falling back to .keys() for older V3 alphas or V2 compatibility
                    try:
                        members = obj.members().items() if hasattr(obj, "members") else obj.items()
                        for key, child in members:
                            find_arrays(child, f"{current_path}/{key}".strip("/"))
                    except Exception:
                        # Final fallback: just iterate keys
                        for key in obj.keys():
                            find_arrays(obj[key], f"{current_path}/{key}".strip("/"))

            find_arrays(group)

        except Exception as e:
            print(f" Error processing {z_path.name}: {e}")


if __name__ == "__main__":
    # Input your path (e.g., D:/files/nd2_files)
    #target = input("Enter the path to the parent folder: ").strip().strip('"').strip("'")
    target = r"D:/files/nd2_files"
    inspect_zarr_v3(target)