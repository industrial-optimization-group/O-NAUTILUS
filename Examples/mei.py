from onautilus.preferential_func_eval import preferential_func_eval as pfe
import numpy as np
from optproblems.zdt import ZDT1
import pandas as pd
from desdeo_problem.Problem import DataProblem
from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)


x = np.random.rand(250, 30)
x_names = ["x"+str(i) for i in range(x.shape[1])]
func = ZDT1()
y = np.asarray([func(x_dat) for x_dat in x])
y_names = ["y"+str(i) for i in range(y.shape[1])]
data = np.hstack((x, y))
data = pd.DataFrame(data, columns=x_names+y_names)

kernel = Matern(nu=3/2)

problem = DataProblem(data=data, variable_names=x_names, objective_names=y_names)
problem.train(GaussianProcessRegressor, {"kernel": kernel})
preference = np.asarray((0.7, 0.7))
result = pfe(problem=problem, reference_point=preference)
print()