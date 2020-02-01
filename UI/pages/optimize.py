from flask import session

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly

from pygmo import fast_non_dominated_sorting as nds
import plotly.graph_objects as go

from UI.app import app

from desdeo_emo.EAs.RVEA import RVEA, oRVEA, robust_RVEA


optimizers = {"RVEA": RVEA, "Optimistic RVEA": oRVEA, "Robust RVEA": robust_RVEA}


def layout():
    return html.Div(
        [
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
    [State("optimization_algorithm", "value")],
)
def optimize(clicked, chosen_algorithm):
    if clicked is None:
        raise PreventUpdate
    problem = session["problem"]
    original_data_y = session["original_dataset"][problem.objective_names].values
    evolver = optimizers[chosen_algorithm](problem, use_surrogates=True)
    while evolver.continue_evolution():
        evolver.iterate()
    objectives = evolver.population.objectives
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
    individuals = evolver.population.individuals
    ##### TRUE FUNCTION EVALUATIONS; REPLACE THIS
    # TODO
    y1 = individuals.sum(axis=1)
    y2 = individuals[:, 0] * individuals[:, 1] - individuals[:, 2]
    figure.add_trace(
        go.Scatter(
            x=y1,
            y=y2,
            mode="markers",
            name="Front from surrogates, evaluated with true functions",
        )
    )
    return (False, "Hi", dcc.Graph(figure=figure, id="graph"))
