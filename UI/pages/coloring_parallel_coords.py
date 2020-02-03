from flask import session

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as ex
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralBiclustering, SpectralCoclustering

from UI.app import app


coloring_strategies = [
    "None",
    "L1 distance from ideal point",
    "L2 distance from ideal point",
    "L-inf distance from ideal point",
    "Spectral Biclustering",
    "Spectral Coclustering",
]


def layout():
    return html.Div(
        [
            html.H1("Parallel Cordinates Plot", id="header_colouring_parallel_coords"),
            html.Label(
                [
                    "Choose the colouring strategy",
                    dcc.Dropdown(
                        id="colouring_strategy",
                        options=[
                            {"label": name, "value": name, "disabled": False}
                            for name in coloring_strategies
                        ],
                        placeholder="Choose the colouring strategy",
                        multi=False,
                        value="None",
                    ),
                ]
            ),
            html.H3(children="", id="graph_title"),
            dcc.Graph(id="graph", figure=[]),
        ]
    )


@app.callback(
    [Output("graph", "figure"), Output("graph_title", "children")],
    [Input("colouring_strategy", "value")],
)
def color_parallel_coords(colouring_strategy):
    if colouring_strategy is None:
        raise PreventUpdate
    data = session["original_dataset"][session["objective_names"]]
    title = "Given data coloured according to "
    if colouring_strategy == "None":
        msg = "No colouring strategy"
        return (ex.parallel_coordinates(data), msg)
    elif colouring_strategy[0] == "L":
        norm_type = colouring_strategy.split()[0][1:]
        return (norm_colouring(data, norm_type), title + colouring_strategy)
    elif colouring_strategy == "Spectral Biclustering":
        return (bicluster_colouring(data), title + colouring_strategy)
    elif colouring_strategy == "Spectral Coclustering":
        return (cocluster_colouring(data), title + colouring_strategy)


def norm_colouring(data, norm_type):
    translated_data = data - data.min(axis=0)
    if norm_type == "1" or norm_type == "2":
        norm = np.linalg.norm(translated_data.values, axis=1, ord=int(norm_type))
    else:
        norm = np.linalg.norm(translated_data.values, axis=1, ord=np.inf)
    norm = pd.Series(norm, name="Distance", index=data.index)
    return ex.parallel_coordinates(data.join(norm), color="Distance")


def bicluster_colouring(data):
    model = SpectralBiclustering(n_clusters=data.shape[1])
    model.fit(data)
    clusters = pd.Series(model.row_labels_, name="Bicluster", index=data.index)
    return ex.parallel_coordinates(
        data.join(clusters),
        dimensions=data.columns[np.argsort(model.column_labels_)].tolist()
        + ["Bicluster"],
        color="Bicluster",
    )


def cocluster_colouring(data):
    model = SpectralCoclustering(n_clusters=data.shape[1])
    model.fit(data)
    clusters = pd.Series(model.row_labels_, name="Bicluster", index=data.index)
    return ex.parallel_coordinates(
        data.join(clusters),
        dimensions=data.columns[np.argsort(model.column_labels_)].tolist()
        + ["Bicluster"],
        color="Bicluster",
    )
