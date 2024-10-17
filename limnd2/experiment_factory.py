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
    frame_count: int
    time_delta: float

    def __init__(self, frame_count: int, time_delta: float):
        self.frame_count = frame_count
        self.time_delta = time_delta
    
    def create_experiment_level(self) -> ExperimentLevel:
        return ExperimentLevel(eType = ExperimentLoopType.eEtTimeLoop,
                               uLoopPars = ExperimentTimeLoop(uiCount=self.frame_count, dPeriod=float(self.time_delta)))
    
    def __str__(self) -> str:
        return f"_T{self.frame_count}"

class NETExp(Exp):
    frame_count: int
    periods: list[tuple[int, float]]

    def __init__(self, periods: list[tuple[int, float]]):
        self.periods = periods
        self.frame_count = sum(t[0] for t in periods)
    
    def create_experiment_level(self) -> ExperimentLevel:
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
    frame_count: int
    stack_delta: float

    def __init__(self, frame_count: int, stack_delta: float):
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
    frame_count: int
    xcoords: list[float]
    ycoords: list[float]
    
    def __init__(self, frame_count: int, xcoords: list[float] = None, ycoords: list[float] = None):

        if xcoords == None:
            xcoords = [0.0 for _ in range(frame_count)]
        
        if ycoords == None:
            ycoords = [0.0 for _ in range(frame_count)]
        
        if not (frame_count == len(xcoords) and frame_count == len(ycoords)):
            raise ValueError("Number of multipoints must match length of both lists with coordinates.")

        self.frame_count = frame_count
        self.xcoords = xcoords
        self.ycoords = ycoords
    
    def create_experiment_level(self):
        points = []
        for xcoord, ycoord in zip(self.xcoords, self.ycoords):
            points.append(ExperimentXYPosLoopPoint(dPosX=float(xcoord), 
                                                   dPosY=float(ycoord)))
            
        return ExperimentLevel(uLoopPars=ExperimentXYPosLoop(uiCount=self.frame_count, Points=points),
                               eType = ExperimentLoopType.eEtXYPosLoop)

    def __str__(self) -> str:
        return f"_M{self.frame_count}"
    
def create_experiment(*args: Exp) -> ExperimentLevel:
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
    
