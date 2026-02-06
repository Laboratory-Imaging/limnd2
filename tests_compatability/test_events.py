from pathlib import Path

import nd2
import numpy as np
import pytest

DATA = Path(__file__).parent / "data"


EXPECTED_COORDS = ([0] * 5 + [1000] * 5 + [2000] * 5) * 3


@pytest.mark.parametrize("fname", ["t3p3z5c3.nd2", "t3p3c3z5.nd2", "t1t1t1p3c3z5.nd2"])
def test_events(fname: str) -> None:
    with nd2.ND2File(DATA / fname) as f:
        events = f.events(orient="list")
        assert isinstance(events, dict)
        if "X Coord [µm]" in events:
            assert len(events["X Coord [µm]"]) > 0
        if "Y Coord [µm]" in events:
            assert len(events["Y Coord [µm]"]) > 0


@pytest.mark.parametrize("fname", ["t3p3z5c3.nd2", "t3p3c3z5.nd2", "t1t1t1p3c3z5.nd2"])
def test_events_pandas(fname: str) -> None:
    pd = pytest.importorskip("pandas")
    with nd2.ND2File(DATA / fname) as f:
        df = pd.DataFrame(f.events())
        assert not df.empty
