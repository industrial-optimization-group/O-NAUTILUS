from flask import session

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from UI.app import app

from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from desdeo_problem.surrogatemodels.lipschitzian import LipschitzianRegressor
from desdeo_emo.EAs.RVEA import RVEA, oRVEA, robust_RVEA

regressors = {
    "Gaussian Process Regressor": GaussianProcessRegressor,
    "Lipschitzian Regressor": LipschitzianRegressor,
}

optimizers = {"RVEA": RVEA, "Optimistic RVEA": oRVEA, "Robust RVEA": robust_RVEA}


def layout():
    return html.Div(
        [
            html.Label(
                [
                    "Choose the surrogate modelling technique",
                    dcc.Dropdown(
                        id="surrogate_modelling_technique",
                        options=[
                            {"label": regressor, "value": regressor}
                            for regressor in regressors
                        ],
                        value="Gaussian Process Regressor",
                    ),
                ]
            ),
            html.Label(
                [
                    "Choose the optimization algorithm",
                    dcc.Dropdown(
                        id="optimization_algorithm",
                        options=[
                            {"label": optimizer, "value": optimizer}
                            for optimizer in optimizers
                        ],
                        value="Optimistic RVEA",
                    ),
                ]
            ),
        ]
    )
