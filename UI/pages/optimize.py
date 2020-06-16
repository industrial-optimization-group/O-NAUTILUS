from flask import session

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np

from pygmo import fast_non_dominated_sorting as nds
import plotly.graph_objects as go
import plotly.express as ex

from UI.app import app

from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.NSGAIII import NSGAIII


optimizers = {"RVEA": RVEA, "NSGAIII": NSGAIII}
selection_type = {
    "Optimistic": {"selection_type": "optimistic"},
    "Mean": {"selection_type": "mean"},
    "Robust": {"selection_type": "robust"},
}
optimizer_hyperparameters = {"n_iterations": 10, "n_gen_per_iter": 20}


def layout():
    return html.Div(
        [
            dbc.Row(
                dbc.Col(
                    html.H1("Optimization", id="header_optimize"),
                    className="row justify-content-center",
                )
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Label(
                            [
                                "Choose the optimization algorithm",
                                dcc.Dropdown(
                                    id="optimization_algorithm",
                                    options=[
                                        {"label": optimizer, "value": optimizer}
                                        for optimizer in optimizers
                                    ],
                                    value="RVEA",
                                ),
                            ]
                        ),
                        width=2,
                    ),
                    dbc.Col(
                        html.Label(
                            [
                                "Choose the selection type",
                                dcc.Dropdown(
                                    id="selection_type",
                                    options=[
                                        {"label": sel_tpye, "value": sel_tpye}
                                        for sel_tpye in selection_type
                                    ],
                                    value="Optimistic",
                                ),
                            ]
                        ),
                        width=2,
                    ),
                ],
                justify="center",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Button(
                        "Optimize problem",
                        id="optimize_button",
                        className="mr-1 mt-1",
                        color="primary",
                    ),
                    className="row justify-content-center",
                )
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Loading(
                        [
                            html.Div(
                                id="optimization_graph_div",
                                hidden=True,
                                children=[html.Div(id="optimization_graph")],
                            )
                        ]
                    ),
                    className="row justify-content-center",
                )
            ),
        ]
    )


@app.callback(
    [
        Output("optimization_graph_div", "hidden"),
        Output("optimization_graph", "children"),
    ],
    [Input("optimize_button", "n_clicks")],
    [State("optimization_algorithm", "value"), State("selection_type", "value")],
)
def optimize(clicked, chosen_algorithm, chosen_selection_type):
    if clicked is None:
        raise PreventUpdate
    problem = session["problem"]
    original_data_y = session["original_dataset"][problem.objective_names].values
    optimizer = optimizers[chosen_algorithm](
        problem,
        use_surrogates=True,
        **optimizer_hyperparameters,
        **selection_type[chosen_selection_type],
    )
    while optimizer.continue_evolution():
        optimizer.iterate()

    session["optimizer"] = optimizer

    fitness_modifier = {"Mean": 0, "Optimistic": -1, "Robust": 1}
    individuals = optimizer.population.individuals
    objectives = (
        optimizer.population.objectives
        + fitness_modifier[chosen_selection_type] * optimizer.population.uncertainity
    )
    fitness = (
        optimizer.population.fitness
        + fitness_modifier[chosen_selection_type] * optimizer.population.uncertainity
    )
    non_dom_indices = nds(fitness)[0][0]
    individuals = individuals[non_dom_indices]
    objectives = objectives[non_dom_indices]
    optimistic_data = pd.DataFrame(
        np.hstack((individuals, objectives)),
        columns=problem.variable_names + problem.objective_names,
    )
    session["optimistic_data"] = optimistic_data
    # plotting
    true_func_eval = session["true_function"]
    known_front = pd.DataFrame(
        original_data_y[nds(original_data_y * problem._max_multiplier)[0][0]],
        columns=problem.objective_names,
    )
    optimistic_front = optimistic_data[problem.objective_names]
    optimistic_front_evaluated = pd.DataFrame(
        true_func_eval(individuals), columns=problem.objective_names
    )
    known_front["Source"] = "Known data"
    optimistic_front["Source"] = "Optimistic front"
    optimistic_front_evaluated["Source"] = "Optimistic front after evaluation"
    data = known_front.append(optimistic_front, ignore_index=True).append(
        optimistic_front_evaluated, ignore_index=True
    )
    figure = ex.scatter_matrix(data, dimensions=problem.objective_names, color="Source")
    return (False, dcc.Graph(figure=figure, id="graph"))
