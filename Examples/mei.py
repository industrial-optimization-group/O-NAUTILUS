from onautilus.preferential_func_eval import preferential_func_eval as pfe
import numpy as np
from optproblems.zdt import ZDT2
import pandas as pd
from desdeo_problem.Problem import DataProblem
from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
    DotProduct,
    ConstantKernel,
)
from desdeo_emo.EAs.RVEA import RVEA, oRVEA, robust_RVEA
from plotly.offline.offline import plot

import plotly.graph_objs as go
import plotly.express as ex


x = np.random.rand(100, 30)
x_names = ["x" + str(i) for i in range(x.shape[1])]
func = ZDT2()
y = np.asarray([func(x_dat) for x_dat in x])
y_names = ["y" + str(i) for i in range(y.shape[1])]
data = np.hstack((x, y))
data = pd.DataFrame(data, columns=x_names + y_names)
filename = "ZDT2_mei_mean.html"
optimizer = RVEA

kernel = Matern(nu=3 / 2)

figure = ex.scatter(x=y[:, 0], y=y[:, 1])
figure.add_scatter(x=[0], y=[0], name="ideal")
figure.add_scatter(x=[1], y=[1], name="nadir")

problem = DataProblem(data=data, variable_names=x_names, objective_names=y_names)
for i in range(30):
    problem.variables[i]._Variable__lower_bound = 0
    problem.variables[i]._Variable__upper_bound = 1
problem.train(GaussianProcessRegressor, {"kernel": kernel})
ea = optimizer(problem, use_surrogates=True)
while ea.continue_evolution():
    ea.iterate()
objectives = ea.population.objectives
evaluated = [func(x) for x in ea.population.individuals]
evaluated = np.asarray(evaluated)
figure.add_scatter(
    x=objectives[:, 0], y=objectives[:, 1], name="optimistic", mode="markers"
)
figure.add_scatter(
    x=evaluated[:, 0], y=evaluated[:, 1], name="evaluated", mode="markers"
)
preference = np.asarray((0.2, 3))
x_new = pfe(problem=problem, reference_point=preference).result.xbest
y_new_approx = problem.evaluate(x_new, use_surrogate=True).objectives
y_new = func(x_new)

figure.add_scatter(
    x=[preference[0]],
    y=[preference[1]],
    name="preference",
    marker={"symbol": "diamond", "size": 20},
)
figure.add_scatter(
    x=[y_new[0]],
    y=[y_new[1]],
    name="New point after evaluation",
    marker={"symbol": "diamond", "size": 20},
)
figure.add_scatter(
    x=[y_new_approx[:, 0][0]],
    y=[y_new_approx[:, 1][0]],
    name="New point before evaluation",
    marker={"symbol": "diamond", "size": 20},
)
x = np.vstack((x, x_new))
y = np.vstack((y, y_new))
data = np.hstack((x, y))
data = pd.DataFrame(data, columns=x_names + y_names)


figure["layout"]["sliders"] = [
    {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Iteration:",
            "visible": True,
            "xanchor": "right",
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }
]

i = -1
frame = {"data": [], "name": str(i)}
sliders_dict = figure["layout"]["sliders"][0]
frame["data"].extend(figure.data)
figure["frames"] += (frame,)
slider_step = {
    "args": [
        [i],
        {
            "frame": {"duration": 300, "redraw": True},
            "mode": "immediate",
            "transition": {"duration": 300},
        },
    ],
    "label": i,
    "method": "animate",
}
sliders_dict["steps"] += (slider_step,)
figure["layout"]["sliders"] = [sliders_dict]
plot(figure, filename=filename)

for i in range(50):
    print(f"Running iteration: {i}")
    fig = ex.scatter(x=y[:, 0], y=y[:, 1])
    fig.add_scatter(x=[0], y=[0], name="ideal")
    fig.add_scatter(x=[1], y=[1], name="nadir")
    problem = DataProblem(data=data, variable_names=x_names, objective_names=y_names)
    for j in range(30):
        problem.variables[j]._Variable__lower_bound = 0
        problem.variables[j]._Variable__upper_bound = 1
    problem.train(GaussianProcessRegressor, {"kernel": kernel})
    ea = optimizer(problem, use_surrogates=True)
    while ea.continue_evolution():
        ea.iterate()
    objectives = ea.population.objectives
    evaluated = [func(x) for x in ea.population.individuals]
    evaluated = np.asarray(evaluated)
    fig.add_scatter(
        x=objectives[:, 0], y=objectives[:, 1], name="optimistic", mode="markers"
    )
    fig.add_scatter(
        x=evaluated[:, 0], y=evaluated[:, 1], name="evaluated", mode="markers"
    )
    x_new = pfe(problem=problem, reference_point=preference).result.xbest
    y_new_approx = problem.evaluate(x_new, use_surrogate=True).objectives
    y_new = func(x_new)

    fig.add_scatter(
        x=[preference[0]],
        y=[preference[1]],
        name="preference",
        marker={"symbol": "diamond", "size": 20},
    )
    fig.add_scatter(
        x=[y_new[0]],
        y=[y_new[1]],
        name="New point after evaluation",
        marker={"symbol": "diamond", "size": 20},
    )
    fig.add_scatter(
        x=[y_new_approx[:, 0][0]],
        y=[y_new_approx[:, 1][0]],
        name="New point before evaluation",
        marker={"symbol": "diamond", "size": 20},
    )
    x = np.vstack((x, x_new))
    y = np.vstack((y, y_new))
    data = np.hstack((x, y))
    data = pd.DataFrame(data, columns=x_names + y_names)

    frame = {"data": [], "name": str(i)}
    sliders_dict = figure["layout"]["sliders"][0]
    frame["data"].extend(fig.data)
    figure["frames"] += (frame,)
    slider_step = {
        "args": [
            [i],
            {
                "frame": {"duration": 300, "redraw": True},
                "mode": "immediate",
                "transition": {"duration": 300},
            },
        ],
        "label": i,
        "method": "animate",
    }
    sliders_dict["steps"] += (slider_step,)
    figure["layout"]["sliders"] = [sliders_dict]
    plot(figure, auto_open=False, filename=filename)

