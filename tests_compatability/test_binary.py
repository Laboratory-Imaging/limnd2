from pathlib import Path

import nd2
import numpy as np
import numpy.testing as npt

DATA = Path(__file__).parent / "data"

# fmt: off
ROW0 = [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,2,2,2,0,0,0,3,0,0,0,0,0,0,0]
# fmt: on


def test_binary():
    with nd2.ND2File(DATA / "with_binary_and_rois.nd2") as f:
        binlayers = f.binary_data
        repr(binlayers)
        repr(binlayers[0])
        assert binlayers is not None
        assert len(binlayers) >= 1
        assert len(binlayers[0]) == f.attributes.sequenceCount
        # you can also index a BinaryLayer directly
        assert isinstance(binlayers[0][2], (np.ndarray, type(None)))
        ary = np.asarray(binlayers)
        assert ary.ndim >= 3
