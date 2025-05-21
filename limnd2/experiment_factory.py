"""
This module contains helper classes and functions for creating
`ExperimentLevel` instances using simplified parameters for each experiment type.
Those instances should be used with `Nd2Writer` instance for altering / creating `.nd2` files.

!!! info
    Since this module is used to creating experiment data structures,
    you should not use any part of this module if you only read an `.nd2` file.

For creating experiments, you should use [`ExperimentFactory`](experiment_factory.md#limnd2.experiment_factory.ExperimentFactory) class.
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from .experiment import ExperimentLevel, ExperimentTimeLoop, ExperimentNETimeLoop, ExperimentXYPosLoop, ExperimentXYPosLoopPoint, ExperimentZStackLoop, ExperimentLoopType, ZStackType
from typing import Any

@dataclass
class _Exp(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError()

    @abstractmethod
    def _create_experiment_level(self) -> ExperimentLevel:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

@dataclass
class _TExp(_Exp):
    """
    Data structure for creating Timeloop experiment instance using number of frames and time between frames.
    """
    count: int = 0
    step: float = 0.0

    def _create_experiment_level(self) -> ExperimentLevel:
        """
        Creates ExperimentLevel instance from simplified settings.
        """
        return ExperimentLevel(eType = ExperimentLoopType.eEtTimeLoop,
                               uLoopPars = ExperimentTimeLoop(uiCount = self.count,
                                                              dPeriod = float(self.step),
                                                              dDuration = (self.count - 1) * float(self.step)))

    def __str__(self) -> str:
        return f"_T{self.count}"
'''
class _NETExp(_Exp):
    """
    Data structure for creating non-equidistant timeloop experiment from list of periods.
    """
    count: int
    periods: list[tuple[int, float]]

    def __init__(self, periods: list[tuple[int, float]]):
        """
        Parameters
        ----------
        periods : list[tuple[int, float]]
            list of periods, each period is a pair made of number of frames in given period and time delta in given period
        """
        self.periods = periods
        self.count = sum(t[0] for t in periods)

    def create_experiment_level(self) -> ExperimentLevel:
        """
        Creates ExperimentLevel instance from simplified settings.
        """
        stages = []
        for count, time_delta in self.periods:
            stages.append(ExperimentTimeLoop(uiCount=count, dPeriod=float(time_delta)))

        loop = ExperimentNETimeLoop(uiCount = self.count,
                                    uiPeriodCount = len(stages),
                                    pPeriodValid = b"\x01" * len(stages),
                                    pPeriod = stages
                                    )

        return ExperimentLevel(uLoopPars=loop,
                               eType = ExperimentLoopType.eEtNETimeLoop)

    def __str__(self) -> str:
        return f"_NET{self.count}"
'''

@dataclass
class _ZExp(_Exp):
    """
    Data structure for creating z-stack experiment using number of frames and distance between frames.
    """
    count: int = 0
    step: float = 0.0
    start: float = None
    end: float = None

    def _create_experiment_level(self) -> ExperimentLevel:
        if self.start is None and self.end is None:
            self.start = 0.0
            self.end = float(self.step * (self.count - 1))
        elif self.start is None:
            self.start = self.end - float(self.step * (self.count - 1))
        elif self.end is None:
            self.end = self.start + float(self.step * (self.count - 1))


        loop = ExperimentZStackLoop(uiCount = self.count,
                                    dZStep = float(self.step),
                                    dZLow = float(self.start),
                                    dZHigh = float(self.end),
                                    bAbsolute = True,
                                    dZHome = float(self.start + (self.end - self.start) / 2)
                                    )


        return ExperimentLevel(eType = ExperimentLoopType.eEtZStackLoop, uLoopPars=loop)

    def __str__(self) -> str:
        return f"_Z{self.count}"

@dataclass
class _MExp(_Exp):
    """
    Data structure for creating multipoint experiment using list of x and y coordinates.
    """
    count: int = 0
    xcoords: list[float] = field(default_factory = list)
    ycoords: list[float] = field(default_factory = list)

    def addPoint(self, x: float, y: float):
        self.xcoords.append(x)
        self.ycoords.append(y)
        self.count += 1

    def _create_experiment_level(self) -> ExperimentLevel:
        """
        Creates ExperimentLevel instance from simplified settings.
        """

        if len(self.xcoords) == 0 and len(self.ycoords) == 0:
            self.xcoords = [0.0] * self.count
            self.ycoords = [0.0] * self.count

        if not (self.count == len(self.xcoords) and self.count == len(self.ycoords)):
            raise ValueError("Number of multipoints must match length of both lists with coordinates.")

        points = []
        for xcoord, ycoord in zip(self.xcoords, self.ycoords):
            points.append(ExperimentXYPosLoopPoint(dPosX=float(xcoord),
                                                   dPosY=float(ycoord)))

        return ExperimentLevel(uLoopPars=ExperimentXYPosLoop(uiCount=self.count, Points=points),
                               eType = ExperimentLoopType.eEtXYPosLoop)

    def __str__(self) -> str:
        return f"_M{self.count}"

def _create_experiment(*args: _Exp) -> ExperimentLevel:
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
        return None

    first = args[0]._create_experiment_level()
    last = first
    for exp in args[1:]:
        new = exp._create_experiment_level()

        #chain new to last
        object.__setattr__(last, "ppNextLevelEx", [new])
        object.__setattr__(last, "uiNextLevelCount", 1)

        last = new
    return first


class ExperimentFactory:
    """
    Helper class for creating experiments, see examples below on how to create timeloop, multipoint and z-stack experiments either
    directly on factory constructor or by modifying values later.

    To actually create experiment instance, make sure to call either
    [`.createExperiment()`](experiment_factory.md#limnd2.experiment_factory.ExperimentFactory.createExperiment)
    method or use call operator.

    ## Sample usage

    ``` py
    from limnd2.experiment_factory import ExperimentFactory

    # only frame counts
    print(ExperimentFactory(t=10, m=5).createExperiment())
    ```

    ??? example "See example output"
        `
        Timeloop experiment(10 frames, interval: No Delay, duration: Continuous), Multipoint experiment(5 frames,
        point coordinates: [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
        `

    ``` py
    # values from dict
    print(ExperimentFactory(t={"count" : 3, "step": 350}, z={"count" : 5, "step": 150}).createExperiment())
    ```

    ??? example "See example output"
        `
        Timeloop experiment(3 frames, interval: 0:00:00.350, duration: Continuous), Z-Stack experiment(5 frames, step: 150.0)
        `

    ``` py
    # combination
    print(ExperimentFactory(t=3, z={"count" : 5, "step": 150}).createExperiment())
    ```

    ??? example "See example output"
        `
        Timeloop experiment(3 frames, interval: No Delay, duration: Continuous), Z-Stack experiment(5 frames, step: 150.0)
        `

    ``` py
    # create factory and modify it
    fac = ExperimentFactory()
    fac.z.count = 10
    fac.z.step = 100
    print(fac())            # createExperiment is called implicitly
    ```

    ??? example "See example output"
        `
        Z-Stack experiment(10 frames, step: 100.0)
        `

    ``` py
    fac = ExperimentFactory()
    fac.m.addPoint(10, 50)
    fac.m.addPoint(20, 70)
    print(fac())
    ```

    ??? example "See example output"
        `
        Multipoint experiment(2 frames, point coordinates: [10.0, 50.0], [20.0, 70.0])
        `

    ``` py
    # inlined multipoint
    print(ExperimentFactory(t=3, z={"count" : 5, "step": 150}, m={"count" : 3, "xcoords" : [10,20,30], "ycoords" : [40,50,60]})())
    ```

    ??? example "See example output"
        `
        Timeloop experiment(3 frames, interval: No Delay, duration: Continuous), Multipoint experiment(3 frames, point coordinates:
        [10.0, 40.0], [20.0, 50.0], [30.0, 60.0]), Z-Stack experiment(5 frames, step: 150.0)
        `

    ``` py
    fac = ExperimentFactory()
    fac.t.count = 10
    fac.z.step = 100
    print(fac())
    ```
    !!! warning
        In this example Z-Stack experiment will be omitted from output since Z-stack frame count is not set,
        even though Z-stack step property was defined.

    ??? example "See example output"
        `
        Timeloop experiment(10 frames, interval: No Delay, duration: Continuous)
        `
    """
    t: _TExp
    m: _MExp
    z: _ZExp

    def __init__(self, *,
                 t : int | dict[str, Any] | None = None,
                 m : int | dict[str, Any] | None = None,
                 z : int | dict[str, Any] | None = None):

        if t is None:
            self.t = _TExp(0, 0)
        elif isinstance(t, int):
            self.t = _TExp(t, 0)
        elif isinstance(t, dict):
            self.t = _TExp(**t)

        if m is None:
            self.m = _MExp(0)
        elif isinstance(m, int):
            self.m = _MExp(m)
        else:
            self.m = _MExp(**m)

        if z is None:
            self.z = _ZExp(0, 0)
        elif isinstance(z, int):
            self.z = _ZExp(z, 0)
        elif isinstance(z, dict):
            self.z = _ZExp(**z)

    def createExperiment(self) -> ExperimentLevel:
        """
        Create `ExperimentLevel` instance using specified settings.
        """
        exps = []
        if self.t.count:
            exps.append(self.t)
        if self.m.count:
            exps.append(self.m)
        if self.z.count:
            exps.append(self.z)
        return _create_experiment(*exps)

    def __call__(self) -> ExperimentLevel:
        """
        Create `ExperimentLevel` instance using specified settings.
        """
        return self.createExperiment()

