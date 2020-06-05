import dash
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = html.Div(
    [
        dcc.Interval(id="clock", interval=1 * 10, n_intervals=0),
        html.Button(id="pause", children="pause"),
        html.Div(id={"type": "box", "name": 1}, children=0),
        html.Div(id={"type": "box", "name": 2}, children=0),
    ]
)


@app.callback(
    Output({"type": "box", "name": ALL}, "children"),
    [Input("clock", "n_intervals")],
    [State({"type": "box", "name": 1}, "children")],
)
def test(tick, current_val):
    return [current_val + 1, current_val * 2]


@app.callback(
    Output("clock", "disabled"),
    [Input("pause", "n_clicks")],
    [State("clock", "disabled")],
)
def toggle(click, state):
    return not state


app.run_server()
