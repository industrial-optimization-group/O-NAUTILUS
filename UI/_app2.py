import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from flask import session, Flask, g, has_app_context, has_request_context
from flask_session import Session
from plotly.subplots import make_subplots


from onautilus.onautilus import ONAUTILUS

x = np.linspace(0.1, 1, 100)
y = 1 / x

# Constructing app
app = dash.Dash(__name__)

app.server.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.server.config.from_object("config.Config")
app.server.config['DEBUG'] = True
Session(app.server)



# Making method and saving it in session
method = ONAUTILUS(known_data=np.vstack((x, y)).T, optimistic_data=[], num_steps=20)


@app.server.route('/')
def set():
    session["method"] = method
    session["preference request"] = None

"""with server.test_request_context():
    print(has_app_context())
    print(has_request_context())
    session["method"] = method
    session["preference request"] = None"""
output = []
set()

def create_plot(request):
    content = request.content
    bounds = content["data"]
    ideal_nadir = content["dimensions_data"]
    preference_point = content["preference"]
    total_steps = content["total_steps"]
    steps_taken = content["steps_taken"]

    fig = make_subplots(
        rows=ideal_nadir.shape[1],
        cols=1,
        shared_xaxes=True,
        subplot_titles=list(ideal_nadir.columns),
    )
    fig.update_xaxes(title_text="Steps", row=2, col=1)
    fig.update_xaxes(range=[0, total_steps])
    for i in range(ideal_nadir.shape[1]):
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
    for i, objective_name in enumerate(ideal_nadir.columns):
        fig.data[5 * i + 0].x = fig.data[5 * i + 0].x + (steps_taken,)
        fig.data[5 * i + 1].x = fig.data[5 * i + 1].x + (steps_taken,)
        fig.data[5 * i + 2].x = fig.data[5 * i + 2].x + (steps_taken,)
        fig.data[5 * i + 3].x = fig.data[5 * i + 3].x + (steps_taken,)
        fig.data[5 * i + 4].x = fig.data[5 * i + 4].x + (steps_taken,)
        fig.data[5 * i + 0].y = fig.data[5 * i + 0].y + (
            bounds[objective_name]["lower_bound"],
        )
        fig.data[5 * i + 1].y = fig.data[5 * i + 1].y + (
            bounds[objective_name]["upper_bound"],
        )
        fig.data[5 * i + 2].y = fig.data[5 * i + 2].y + (
            preference_point[objective_name][0],
        )
        fig.data[5 * i + 3].y = fig.data[5 * i + 3].y + (
            ideal_nadir[objective_name]["ideal"],
        )
        fig.data[5 * i + 4].y = fig.data[5 * i + 4].y + (
            ideal_nadir[objective_name]["nadir"],
        )
    return fig


def extendplot(request):
    content = request.content
    bounds = content["data"]
    ideal_nadir = content["dimensions_data"]
    preference_point = content["preference"]
    total_steps = content["total_steps"]
    steps_taken = content["steps_taken"]

    extension = [dict(x=[], y=[]), []]
    for i, objective_name in enumerate(ideal_nadir.columns):
        extension[0]["x"].append([steps_taken])
        extension[0]["x"].append([steps_taken])
        extension[0]["x"].append([steps_taken])
        extension[0]["x"].append([steps_taken])
        extension[0]["x"].append([steps_taken])
        extension[0]["y"].append([bounds[objective_name]["lower_bound"]])
        extension[0]["y"].append([bounds[objective_name]["upper_bound"]])
        extension[0]["y"].append([preference_point[objective_name][0]])
        extension[0]["y"].append([ideal_nadir[objective_name]["ideal"]])
        extension[0]["y"].append([ideal_nadir[objective_name]["nadir"]])
        extension[1] = extension[1] + [
            5 * i,
            5 * i + 1,
            5 * i + 2,
            5 * i + 3,
            5 * i + 4,
        ]
    return extension


def createscatter2d(request):
    non_dom = request.content["data"].values
    achievable = non_dom[request.content["achievable_ids"]]
    ideal = request.content["dimensions_data"].loc["ideal"]
    nadir = request.content["dimensions_data"].loc["nadir"]
    current = request.content["current_point"].values[0]
    preference = request.content["preference"].values[0]
    figure = go.Figure()
    figure.update_layout(
        title={
            "text": "X1 = 1/X0",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            #'yanchor': 'top'
        }
    )
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


@app.server.route("/create layout/")
def create_layout():

    first_data = create_plot(session.get("method").request_ranges_plot())
    scattergraph = createscatter2d(session.get("method").request_solutions_plot())
    first_pref = session.get("method").preference_point
    preference_request = session.get("method").request_preferences()

    layout = html.Div(
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
            html.Div(
                [
                    dcc.Input(
                        id=f"textbox{i+1}",
                        placeholder=f"Enter a value for X{i+1}",
                        type="number",
                        value=first_pref[i],  # TODO remove
                        disabled=True,
                    )
                    for i in range(2)
                ]
            ),
            html.Button("Pause", id="pausebutton"),
            html.Button("Submit", id="submitbutton", disabled=True),
            dcc.Interval(
                id="step_counter", interval=1 * 1000, n_intervals=0
            ),  # in milliseconds
            # Hidden div inside the app that stores the intermediate value
            html.Div(id="preference_values_container", style={"display": "none"}),
        ]
    )
    return layout


app.layout = create_layout()


@app.callback(
    [
        Output(component_id="example-graph", component_property="extendData"),
        Output("scatter-graph", "figure"),
        Output("pausebutton", "n_clicks"),
    ],
    [Input(component_id="step_counter", component_property="n_intervals")],
    [State("preference_values_container", "children")],
)
def iterate(n, preference_values):
    ranges_plot_request, solutions_plot_request, preference_request = session.get(
        "method"
    ).iterate(session.get("preference request"))
    session["preference request"] = preference_request
    if session.get("preference request").interaction_priority == "required":
        data_extension = None
        scattergraph = createscatter2d(solutions_plot_request)
        click_pause = 1
    else:
        data_extension = extendplot(ranges_plot_request)
        scattergraph = createscatter2d(solutions_plot_request)
        click_pause = None
    return (data_extension, scattergraph, click_pause)


@app.callback(
    [
        Output("step_counter", "disabled"),
        Output("pausebutton", "children"),
        Output("submitbutton", "disabled"),
        *[Output(f"textbox{i+1}", "disabled") for i in range(2)],
        # *[Output(f"textbox{i+1}", "value") for i in range(2)]
    ],
    [Input("pausebutton", "n_clicks")],
    [State("pausebutton", "children")],
)
def pauseevent(pauseclick, pausevalue):
    if pauseclick is None:
        # Prevents pausing when initializing or other non-pausing events
        return (False, "Pause", True, True, True)
        # return (True, "Play", False, False, False)
    if pausevalue == "Pause":
        return (True, "Play", False, False, False)
    elif pausevalue == "Play":
        return (False, "Pause", True, True, True)


@app.callback(
    Output("preference_values_container", "children"),
    [Input("submitbutton", "n_clicks")],
    [*[State(f"textbox{i+1}", "value") for i in range(2)]],
)
def submitevent(submitclick, *preference_values):
    preference_request = session.get("preference request")
    pref_value = pd.DataFrame(
        [preference_values],
        columns=preference_request.content["dimensions_data"].columns,
    )
    preference_request.response = pref_value
    session["preference request"] = preference_request
    return preference_values


if __name__ == "__main__":
    app.run_server(debug=True)
