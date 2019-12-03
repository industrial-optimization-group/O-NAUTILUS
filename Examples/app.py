import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from onautilus.onautilus import ONAUTILUS
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

x = np.linspace(0.1, 1, 100)
y = 1 / x


method = ONAUTILUS(known_data=np.vstack((x, y)).T, optimistic_data=[], num_steps=20)
method.preference_point = np.asarray([0.2, 4])
output = []


def create_plot(request):
    bounds, _, _, preference_point, current_point, ideal, nadir, steps_taken, total_steps = (
        request
    )
    graphdata = zip(bounds, preference_point, current_point, ideal, nadir)

    fig = make_subplots(
        rows=len(ideal),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"X{i}" for i in range(len(ideal))],
    )
    fig.update_xaxes(title_text="Steps", row=2, col=1)
    fig.update_xaxes(range=[0, total_steps])
    for i in range(len(ideal)):
        legend = True if i == 0 else False
        # lower bound
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name=f"Lower Bound",
                showlegend=False,
                mode="lines",
                line_color="green",
            ),
            row=i + 1,
            col=1,
        )
        # upper bound
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                fill="tonexty",
                name=f"Reachable area",
                mode="lines",
                line_color="green",
                showlegend=legend,
            ),
            row=i + 1,
            col=1,
        )
        # preference
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name=f"Preference",
                showlegend=legend,
                mode="lines",
                line_dash="dash",
                line_color="brown",
            ),
            row=i + 1,
            col=1,
        )
        # ideal point
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name=f"ideal",
                mode="lines",
                line_dash="dash",
                line_color="green",
                showlegend=legend,
            ),
            row=i + 1,
            col=1,
        )
        # nadir point
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name=f"nadir",
                mode="lines",
                line_dash="dash",
                line_color="red",
                showlegend=legend,
            ),
            row=i + 1,
            col=1,
        )

    for i, (bound, preference, current_val, ideal_val, nadir_val) in enumerate(
        graphdata
    ):
        fig.data[5 * i + 0].x = fig.data[5 * i + 0].x + (steps_taken,)
        fig.data[5 * i + 1].x = fig.data[5 * i + 1].x + (steps_taken,)
        fig.data[5 * i + 2].x = fig.data[5 * i + 2].x + (steps_taken,)
        fig.data[5 * i + 3].x = fig.data[5 * i + 3].x + (steps_taken,)
        fig.data[5 * i + 4].x = fig.data[5 * i + 4].x + (steps_taken,)
        fig.data[5 * i + 0].y = fig.data[5 * i + 0].y + (bound[0],)
        fig.data[5 * i + 1].y = fig.data[5 * i + 1].y + (bound[1],)
        fig.data[5 * i + 2].y = fig.data[5 * i + 2].y + (preference,)
        fig.data[5 * i + 3].y = fig.data[5 * i + 3].y + (ideal_val,)
        fig.data[5 * i + 4].y = fig.data[5 * i + 4].y + (nadir_val,)
    return fig


def extendplot(request):
    bounds, _, _, preference_point, current_point, ideal, nadir, steps_taken, total_steps = (
        request
    )
    graphdata = zip(bounds, preference_point, current_point, ideal, nadir)
    extension = [dict(x=[], y=[]), []]
    for i, (bound, preference, current_val, ideal_val, nadir_val) in enumerate(
        graphdata
    ):
        extension[0]["x"].append([steps_taken])
        extension[0]["x"].append([steps_taken])
        extension[0]["x"].append([steps_taken])
        extension[0]["x"].append([steps_taken])
        extension[0]["x"].append([steps_taken])
        extension[0]["y"].append([bound[0]])
        extension[0]["y"].append([bound[1]])
        extension[0]["y"].append([preference])
        extension[0]["y"].append([ideal_val])
        extension[0]["y"].append([nadir_val])
        extension[1] = extension[1] + [
            5 * i,
            5 * i + 1,
            5 * i + 2,
            5 * i + 3,
            5 * i + 4,
        ]
    return extension


def createscatter2d(request):
    non_dom = request[1]
    achievable = request[1][request[2]]
    ideal = request[5]
    nadir = request[6]
    current = request[4]
    preference = request[3]
    figure = go.Figure()
    figure.update_layout(
    title={
        'text': "Y = 1/X",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        #'yanchor': 'top'
        })
    figure.add_trace(
        go.Scatter(
            x=non_dom[:, 0],
            y=non_dom[:, 1],
            name="Non-dominated front",
            line_color="grey",
        )
    )
    figure.add_trace(
        go.Scatter(x=achievable[:, 0], y=achievable[:, 1], name="Achievable front")
    )
    figure.add_trace(go.Scatter(x=[ideal[0]], y=[ideal[1]], name="Ideal point"))
    figure.add_trace(go.Scatter(x=[nadir[0]], y=[nadir[1]], name="Nadir point"))
    figure.add_trace(go.Scatter(x=[current[0]], y=[current[1]], name="Current Point"))
    figure.add_trace(
        go.Scatter(x=[preference[0]], y=[preference[1]], name="Preference")
    )
    return figure


first_data = create_plot(method.requests_plot())
scattergraph = createscatter2d(method.requests_plot())
app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1(children="NAUTILUS test app"),
        html.Div(
            children="""
        Testing out dash plotly
    """
        ),
        html.Div(
            [dcc.Graph(id="example-graph", figure=first_data)],
            style={"width": "49%", "display": "inline-block"},
        ),
        html.Div(
            [dcc.Graph(id="scatter-graph", figure=scattergraph)],
            style={"width": "49%", "display": "inline-block"},
        ),
        dcc.Interval(id="time", interval=1 * 1000, n_intervals=0),  # in milliseconds
    ]
)


@app.callback(
    [
        Output(component_id="example-graph", component_property="extendData"),
        Output(component_id="scatter-graph", component_property="figure"),
    ],
    [Input(component_id="time", component_property="n_intervals")],
)
def iterate(n):
    method.iterate()
    output = method.requests_plot()
    if len(output[2]) == 0:
        return
    else:
        return extendplot(output), createscatter2d(output)


if __name__ == "__main__":
    app.run_server()
