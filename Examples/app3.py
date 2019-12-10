import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

app = dash.Dash()

app.layout = html.Div(
    [
        dcc.Interval(id="interval"),
        html.P(id="output"),
        html.Button("toggle interval", id="button"),
    ]
)


@app.callback(
    [Output("interval", "disabled"),
    Output("output", "children")],
    [Input("button", "n_clicks"), 
    Input("interval", "n_intervals")],
    [State("interval", "disabled")],
)
def toggle_interval(n, n_int, disabled):
    msg = f"Interval has fired {n_int} times"
    if n is None:
        return False, msg
    elif n%2==1:
        return True, msg
    return False, msg



if __name__ == "__main__":
    app.run_server(debug=True)