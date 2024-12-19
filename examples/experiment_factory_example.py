from limnd2.experiment_factory import ExperimentFactory

# only frame counts
print(ExperimentFactory(t=10, m=5).createExperiment())

# values from dict
print(ExperimentFactory(t={"count" : 3, "step": 350}, z={"count" : 5, "step": 150}).createExperiment())

# combination
print(ExperimentFactory(t=3, z={"count" : 5, "step": 150}).createExperiment())

# create factory and modify it
fac = ExperimentFactory()
fac.z.count = 10
fac.z.step = 100
print(fac())            # createExperiment is called implicitly

# multipoints
fac = ExperimentFactory()
fac.m.addPoint(10, 50)
fac.m.addPoint(20, 70)
print(fac())

# inlined multipoint
print(ExperimentFactory(t=3, z={"count" : 5, "step": 150}, m={"count" : 3, "xcoords" : [10,20,30], "ycoords" : [40,50,60]})())

# z set without frame count = zstack will be omitted
fac = ExperimentFactory()
fac.t.count = 10
fac.z.step = 100
print(fac())
