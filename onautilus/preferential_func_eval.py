import cma
from desdeo_problem.Problem import DataProblem
import numpy as np
from scipy.stats import norm, uniform
import pandas as pd
from desdeo_tools.scalarization import PointMethodASF


def preferential_func_eval(
    problem: DataProblem,
    optimistic_front: pd.DataFrame,
    ideal: np.ndarray,
    nadir: np.ndarray,
    reference_point: np.ndarray,
    distribution: str = "Normal",
):
    """Make sure that the dimensions are normalized such that a single "sigma" value
    is meaningful"""
    lower_bounds = problem.get_variable_lower_bounds()
    upper_bounds = problem.get_variable_upper_bounds()
    # Maybe solve ASF with known data to get better initial solution
    """rand = np.random.random(lower_bounds.shape)
    initial_solution = (upper_bounds - lower_bounds) * rand + lower_bounds"""
    initial_solution = asf_func_eval(
        problem, optimistic_front, ideal, nadir, reference_point
    )
    # Solution has to be within 3 sigma of the initial_solution
    sigma = max((upper_bounds - lower_bounds) / 2)
    solver = cma.CMAEvolutionStrategy(
        x0=initial_solution,
        sigma0=sigma,
        inopts={"bounds": [lower_bounds, upper_bounds]},
    )
    return solver.optimize(
        m_EI, args=(problem, reference_point, distribution), verb_disp=0
    )


def asf_func_eval(
    problem: DataProblem,
    optimistic_front: pd.DataFrame,
    ideal: np.ndarray,
    nadir: np.ndarray,
    reference_point: np.ndarray,
):
    # TODO Support maximization!!!
    scalarization_func = PointMethodASF(nadir=nadir, ideal=ideal)
    fitness = optimistic_front[problem.objective_names]
    chosen_id = scalarization_func(fitness, reference_point).argmin()
    chosen_solution = optimistic_front[problem.variable_names].values[chosen_id]
    return chosen_solution


def m_EI(x, problem: DataProblem, reference_point: np.ndarray, distribution: str):
    results = problem.evaluate(x, use_surrogate=True)
    means = results.objectives
    std = results.uncertainity

    mEI = 1
    for i in range(problem.n_of_objectives):
        if distribution == "Normal":
            mEI *= _expected_improvement_normal(
                means[:, i], std[:, 1], reference_point[i]
            )
        elif distribution == "Uniform":
            mEI *= _expected_improvement_normal(
                means[:, i], std[:, 1], reference_point[i]
            )
        else:
            raise ValueError(
                f"Distribution type {distribution} not supported. Please"
                " use 'Normal' or 'Uniform' distribution"
            )
    return -mEI


def _expected_improvement_normal(means, std, y_ref):
    with np.errstate(divide="warn"):
        imp = y_ref - means
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        # ei[sigma == 0.0] = 0.0
    return ei


def _expected_improvement_uniform(means, std, y_ref):
    sigma = 2 * std / np.sqrt(12)
    with np.errstate(divide="warn"):
        imp = y_ref - means
        Z = imp / sigma
        ei = imp * uniform.cdf(Z) + sigma * uniform.pdf(Z)
        # ei[sigma == 0.0] = 0.0
    return ei
