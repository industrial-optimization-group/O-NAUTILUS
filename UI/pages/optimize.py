from flask import session

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly
import numpy as np

from pygmo import fast_non_dominated_sorting as nds
import plotly.graph_objects as go

from UI.app import app

from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.NSGAIII import NSGAIII


optimizers = {"RVEA": RVEA, "NSGAIII": NSGAIII}
selection_type = {
    "Optimistic": {"selection_type": "optimistic"},
    "Mean": {"selection_type": "mean"},
    "Robust": {"selection_type": "robust"},
}
optimizer_hyperparameters = {"n_iterations": 6, "n_gen_per_iter": 50}


def layout():
    return html.Div(
        [
            html.H1("Optimization", id="header_optimize"),
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
            dcc.Loading(
                [
                    html.Button("Optimize problem", id="optimize_button"),
                    html.Div(
                        id="optimization_graph_div",
                        hidden=True,
                        children=[
                            dcc.Markdown(id="optimization_graph_md"),
                            html.Div(id="optimization_graph"),
                        ],
                    ),
                ]
            ),
        ]
    )


@app.callback(
    [
        Output("optimization_graph_div", "hidden"),
        Output("optimization_graph_md", "children"),
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
        **selection_type[chosen_selection_type]
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
    session["optimistic_data"] = pd.DataFrame(
        np.hstack((individuals, objectives)),
        columns=problem.variable_names + problem.objective_names,
    )
    """data = pd.DataFrame(objectives, columns=problem.objective_names)
    fig = ex.parallel_coordinates(data)
    fig_obj = dcc.Graph(figure=fig)"""
    figure = go.Figure()
    figure.update_layout(
        title="Non-dominated fronts",
        xaxis_title=problem.objective_names[0],
        yaxis_title=problem.objective_names[1],
    )
    known_front = original_data_y[nds(original_data_y)[0][0]]
    figure.add_trace(
        go.Scatter(
            x=known_front[:, 0],
            y=known_front[:, 1],
            mode="markers",
            name="Front from known data",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=objectives[:, 0],
            y=objectives[:, 1],
            mode="markers",
            name="Front from surrogates",
        )
    )
    # REPLACE THIS
    # TODO
    true_func_eval = session["true_function"]
    y = true_func_eval(individuals)
    figure.add_trace(
        go.Scatter(
            x=y[:, 0],
            y=y[:, 1],
            mode="markers",
            name="Front from surrogates, evaluated with true functions",
        )
    )
    return (False, "", dcc.Graph(figure=figure, id="graph"))
