"""
This module contains helper classes and functions for creating ExperimentLevel instances using simplified parameters for each experiment type.
Those instances should be used with Nd2Writer instance for altering / creating .nd2 files.

!!! warning
    Since this module is used to creating experiment data structures, you should not use any part of this module if you only read an .nd2 file.
"""


from abc import ABC, abstractmethod
from .experiment import ExperimentLevel, ExperimentTimeLoop, ExperimentNETimeLoop, ExperimentXYPosLoop, ExperimentXYPosLoopPoint, ExperimentZStackLoop, ExperimentLoopType

class Exp(ABC):
    frame_count: int

    @abstractmethod
    def __init__(self):
        raise NotImplementedError()

    @abstractmethod
    def create_experiment_level(self) -> ExperimentLevel:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class TExp(Exp):
    """
    Data structure for creating Timeloop experiment instance using number of frames and time between frames.
    """
    frame_count: int
    time_delta: float

    def __init__(self, frame_count: int, time_delta: float):
        """
        Parameters
        ----------
        frame_count : int
            number of frames in the experiment loop
        time_delta : float
            time between frames in miliseconds
        """
        self.frame_count = frame_count
        self.time_delta = time_delta

    def create_experiment_level(self) -> ExperimentLevel:
        """
        Creates ExperimentLevel instance from simplified settings.
        """
        return ExperimentLevel(eType = ExperimentLoopType.eEtTimeLoop,
                               uLoopPars = ExperimentTimeLoop(uiCount=self.frame_count, dPeriod=float(self.time_delta)))

    def __str__(self) -> str:
        return f"_T{self.frame_count}"

class NETExp(Exp):
    """
    Data structure for creating non-equidistant timeloop experiment from list of periods.
    """
    frame_count: int
    periods: list[tuple[int, float]]

    def __init__(self, periods: list[tuple[int, float]]):
        """
        Parameters
        ----------
        periods : list[tuple[int, float]]
            list of periods, each period is a pair made of number of frames in given period and time delta in given period
        """
        self.periods = periods
        self.frame_count = sum(t[0] for t in periods)

    def create_experiment_level(self) -> ExperimentLevel:
        """
        Creates ExperimentLevel instance from simplified settings.
        """
        stages = []
        for count, time_delta in self.periods:
            stages.append(ExperimentTimeLoop(uiCount=count, dPeriod=float(time_delta)))

        loop = ExperimentNETimeLoop(uiCount = self.frame_count,
                                    uiPeriodCount = len(stages),
                                    pPeriodValid = b"\x01" * len(stages),
                                    pPeriod = stages
                                    )

        return ExperimentLevel(uLoopPars=loop,
                               eType = ExperimentLoopType.eEtNETimeLoop)

    def __str__(self) -> str:
        return f"_NET{self.frame_count}"


class ZExp(Exp):
    """
    Data structure for creating z-stack experiment using number of frames and distance between frames.
    """
    frame_count: int
    stack_delta: float

    def __init__(self, frame_count: int, stack_delta: float):
        """
        Parameters
        ----------
        frame_count : int
            number of frames in the experiment loop
        stack_delta : float
            distance between frames in micrometers
        """
        self.frame_count = frame_count
        self.stack_delta = stack_delta

    def create_experiment_level(self) -> ExperimentLevel:
        loop = ExperimentZStackLoop(uiCount = self.frame_count,
                                    dZStep = float(self.stack_delta),
                                    dZHigh = float(self.stack_delta * self.frame_count))

        return ExperimentLevel(eType = ExperimentLoopType.eEtZStackLoop,
                               uLoopPars=loop)

    def __str__(self) -> str:
        return f"_Z{self.frame_count}"

class MExp(Exp):
    """
    Data structure for creating multipoint experiment using list of x and y coordinates.
    """
    frame_count: int
    xcoords: list[float]
    ycoords: list[float]

    def __init__(self, frame_count: int, xcoords: list[float] = None, ycoords: list[float] = None):
        """
        Parameters
        ----------
        frame_count : int
            number of frames in the experiment loop
        xcoords : list[float] = None
            list of x coordinates
        ycoords : list[float] = None
            list of y coordinates

        Raises
        ------
        ValueError
            if lengths of coordinate lists are not equal to `frame_count`.
        """
        if xcoords == None:
            xcoords = [0.0 for _ in range(frame_count)]

        if ycoords == None:
            ycoords = [0.0 for _ in range(frame_count)]

        if not (frame_count == len(xcoords) and frame_count == len(ycoords)):
            raise ValueError("Number of multipoints must match length of both lists with coordinates.")

        self.frame_count = frame_count
        self.xcoords = xcoords
        self.ycoords = ycoords

    def create_experiment_level(self) -> ExperimentLevel:
        """
        Creates ExperimentLevel instance from simplified settings.
        """
        points = []
        for xcoord, ycoord in zip(self.xcoords, self.ycoords):
            points.append(ExperimentXYPosLoopPoint(dPosX=float(xcoord),
                                                   dPosY=float(ycoord)))

        return ExperimentLevel(uLoopPars=ExperimentXYPosLoop(uiCount=self.frame_count, Points=points),
                               eType = ExperimentLoopType.eEtXYPosLoop)

    def __str__(self) -> str:
        return f"_M{self.frame_count}"

def create_experiment(*args: Exp) -> ExperimentLevel:
    """
    This function chains and nests several simplified experiments into single ExperimentLevel instance.

    !!! warning
        This function chains experiments without any consideration for their validity or order,
        experiments should follow timeloop, multipoint, z-stack order and each experiment type should
        be used at most once, however this function **does not** enforce any of this.

    Parameters
    ----------
    *args : Exp
        list of simplified experiments to chain



    """
    if len(args) < 1:
        raise ValueError("You must provide at least one experiment.")

    first = args[0].create_experiment_level()
    last = first
    for exp in args[1:]:
        new = exp.create_experiment_level()

        #chain new to last
        object.__setattr__(last, "ppNextLevelEx", [new])
        object.__setattr__(last, "uiNextLevelCount", 1)

        last = new
    return first



if __name__ == "__main__":
    Z = ZExp(10, 150)
    T = TExp(5, 100)
    M = MExp(2, [100,200], [200,100])
    NET = NETExp([(5,100), (10, 200), (2, 150)])

