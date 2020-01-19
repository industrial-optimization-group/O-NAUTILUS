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
    [Output("button", "n_clicks"),
    Output("output", "children")],
    [Input("interval", "n_intervals")],
    [State("button", "n_clicks")]

)
def toggle_interval(n_int, n_clicks):
    msg = f"Interval has fired {n_int} times"
    if n_int == 5:
        if n_clicks is None:
            return (0, msg)
        else: 
            return(n_clicks+1, msg)
    return n_clicks, msg

@app.callback(Output("interval", "disabled"), [Input("button", "n_clicks")])
def pauseevent(buttonpressed):
    print(buttonpressed)
    if buttonpressed is not None:
        print(buttonpressed)
        return (True)


if __name__ == "__main__":
    app.run_server(debug=True)