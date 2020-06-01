from dash.exceptions import PreventUpdate
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.express as ex
import plotly.graph_objects as go


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
cols = ["a", "b", "c", "d"]

app.layout = html.Div(
    children=[
        dbc.Row(html.H1("Navigation", id="header_navigator")),
        dbc.Row(
            [
                dbc.Col(html.H3("Aspiration Levels"), width=2),
                dbc.Col(html.H3("Navigator View"), width=9),
            ]
        ),
        html.Div(
            id="bigbox",
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    html.H4(
                                        f"Enter a value for {y}", className="card-title"
                                    ),
                                    dcc.Input(
                                        id=f"textbox{i+1}",
                                        placeholder=f"Enter a value for {y}",
                                        type="number",
                                    ),
                                ]
                            ),
                            width=2,
                        ),
                        dbc.Col(dcc.Graph(id=f"nav_graph{i+1}", figure=None), width=9),
                    ]
                )
                for i, y in enumerate(cols)
            ],
        ),
        html.Button("Pause", id="pausebutton"),
        html.Div("hole", id="hole"),
    ]
)


@app.callback(
    Output("hole", "children"),
    [Input("pausebutton", "n_clicks")],
    [State("bigbox", "children")],
)
def test(click, stuff):
    print(stuff)
    return 1


if __name__ == "__main__":
    app.run_server(debug=True)
