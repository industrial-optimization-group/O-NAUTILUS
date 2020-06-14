import dash
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL
import plotly.express as ex
import numpy as np
import pandas as pd

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
data = pd.DataFrame(np.random.rand(100, 3), columns=["a", "s", "d"])

app.layout = html.Div(
    [
        dbc.Row(dbc.Col(dbc.Button(id="button"))),
        dbc.Row(
            dbc.Col(
                dbc.Collapse(
                    id="graph",
                    children=[
                        dbc.Row(dbc.Col(html.H1("hi"))),
                        dbc.Row(
                            dbc.Col([dcc.Graph(figure=ex.parallel_coordinates(data))])
                        ),
                    ],
                    is_open=True,
                )
            )
        ),
        html.Div("hi", hidden=True),
    ]
)


@app.callback(
    Output("graph", "is_open"),
    [Input("button", "n_clicks")],
    [State("graph", "is_open")],
)
def collapse(press, state):
    return not state


app.run_server(debug=True)
