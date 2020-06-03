from flask import session

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import numpy as np

from UI.app import app

from desdeo_problem.Problem import DataProblem
from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from desdeo_problem.surrogatemodels.lipschitzian import LipschitzianRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import r2_score

regressors = {
    "Kriging": GaussianProcessRegressor,
    "Lipschitzian Regressor": LipschitzianRegressor,
}

regressor_hyperparameters = {
    "Kriging": {"kernel": Matern(nu=3 / 2)},
    "Lipschitzian Regressor": None,
}


def layout():
    objective_names = session["objective_names"]
    return html.Div(
        [
            # Header
            dbc.Row(
                dbc.Col(
                    html.H1("Surrogate Modelling", id="header_train_model"),
                    className="row justify-content-center",
                )
            ),
            # Modelling technique dropdown
            dbc.Row(
                dbc.Col(
                    html.Label(
                        [
                            "Choose the surrogate modelling technique",
                            dcc.Dropdown(
                                id="surrogate_modelling_technique",
                                options=[
                                    {"label": regressor, "value": regressor}
                                    for regressor in regressors
                                ],
                                value="Kriging",
                            ),
                        ]
                    ),
                    className="row justify-content-center",
                )
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Loading([dbc.Button("Train models", id="train_models")]),
                    className="row justify-content-center",
                )
            ),
            html.Div(
                id="results_selection_train_models",
                hidden=True,
                children=dcc.Dropdown(
                    id="results_selection_train_models_dropdown",
                    options=[
                        {"label": objective, "value": objective}
                        for objective in objective_names
                    ],
                ),
            ),
            html.Div(
                id="results_train_model",
                hidden=True,
                children=[
                    dcc.Markdown(id="results_text_train_models"),
                    html.Div(id="results_graph_train_models"),
                ],
            ),
        ]
    )


@app.callback(
    [
        Output("results_train_model", "hidden"),
        Output("results_selection_train_models", "hidden"),
        Output("results_selection_train_models_dropdown", "value"),
    ],
    [Input("train_models", "n_clicks")],
    [State("surrogate_modelling_technique", "value")],
)
def train_all_models(button_clicked, chosen_technique):
    if button_clicked is None:
        raise PreventUpdate
    regressor = regressors[chosen_technique]
    objective_names = session["objective_names"]
    variable_names = session["decision_variable_names"]
    data = session["original_dataset"]
    bounds = session["bounds"]
    maximize = session["maximization_info"]
    problem = DataProblem(
        data=data,
        objective_names=objective_names,
        variable_names=variable_names,
        bounds=bounds,
        maximize=maximize,
    )
    # Use the following what analytical problems are supported.
    # Get problem from previous page instead of creating one here.

    # problem = session["problem"]
    problem.train(regressor, regressor_hyperparameters[chosen_technique])
    session["problem"] = problem
    """for objective in objective_names:
        model = regressor()
        model.fit(data[variable_names].values, data[objective].values)
        session[objective + "_model"] = model"""
    return [False, False, objective_names[0]]


@app.callback(
    [
        Output("results_text_train_models", "children"),
        Output("results_graph_train_models", "children"),
    ],
    [Input("results_selection_train_models_dropdown", "value")],
)
def show_results(objective):
    if objective is None:
        raise PreventUpdate
    data = session["original_dataset"]
    variable_names = session["decision_variable_names"]
    # model = session[objective + "_model"]
    problem = session["problem"]
    y_true = data[objective]
    # y_pred_mean, y_pred_std = model.predict(data[variable_names].values)
    y_pred_mean_all = problem.evaluate(
        data[variable_names].values, use_surrogate=True
    ).objectives
    y_pred_mean = y_pred_mean_all[:, problem.objective_names.index(objective)]
    r2_value = r2_score(y_true, y_pred_mean)
    r2_message = f"R2 score for model on objective {objective}: {r2_value}"
    figure = go.Figure()
    figure.update_layout(
        title="True objective values vs Predicted objective values",
        xaxis_title="True objective values",
        yaxis_title="Predicted objective values",
    )
    figure.add_trace(
        go.Scatter(x=y_true, y=y_pred_mean, mode="markers", name="Predictions")
    )
    perfectline = np.linspace(np.min(y_true), np.max(y_true), 100)
    figure.add_trace(
        go.Scatter(x=perfectline, y=perfectline, mode="lines", name="45 Degree line")
    )
    return [r2_message, dcc.Graph(figure=figure, id="graph")]
