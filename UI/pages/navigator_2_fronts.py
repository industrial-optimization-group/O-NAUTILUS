from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd

from dash.dependencies import Input, Output, State
from flask import session


from onautilus.onautilus import ONAUTILUS
from UI.app import app
from UI.pages.tools import create_navigator_plot, extend_navigator_plot, createscatter2d


def layout():
    return html.Div(
        children=[
            html.H1(children="NAUTILUS test app"),
            html.Div(
                children="""
            Testing out dash plotly
        """
            ),
            html.Div(
                id="navigator_graph",
                style={"width": "49%", "display": "inline-block"},
                children=dcc.Graph(id="navigator_graph_figure"),
            ),
            html.Div(
                id="scatter_graph", style={"width": "49%", "display": "inline-block"}
            ),
            html.Div(
                [
                    dcc.Input(
                        id=f"textbox{i+1}",
                        placeholder=f"Enter a value for X{i+1}",
                        type="number",
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
            html.Div(id="callback_blackhole"),
        ]
    )


@app.callback(
    [
        Output(component_id="navigator_graph_figure", component_property="figure"),
        Output("scatter_graph", "children"),
        Output("pausebutton", "n_clicks"),
    ],
    [Input(component_id="step_counter", component_property="n_intervals")],
    [State(component_id="navigator_graph_figure", component_property="figure")],
)
def iterate(n, navigator_graph):
    if n is None:
        raise PreventUpdate
    # Zeroth iteration
    if n == 0:
        known_data = session["original_dataset"][session["objective_names"]].values
        surrogate_data = session["optimizer"].population.objectives
        method = ONAUTILUS(
            known_data=known_data, optimistic_data=surrogate_data, num_steps=20
        )
        session["navigator"] = method
        ranges_plot_request, solutions_plot_request, preference_request = (
            method.requests()
        )
        if "preference" in session.keys():
            preference_request.response = session["preference"].response
        session["preference"] = preference_request
        navigator_graph_new = create_navigator_plot(ranges_plot_request)
        scatter_graph = dcc.Graph(figure=createscatter2d(solutions_plot_request))
        return (navigator_graph_new, scatter_graph, 1)
    # Other cases
    else:
        method = session["navigator"]
        preference_request = session["preference"]
        ranges_plot_request, solutions_plot_request, preference_request = method.iterate(
            preference_request
        )
        session["preference"] = preference_request
        if preference_request.interaction_priority == "required":
            navigator_graph_new = navigator_graph
            scatter_graph = dcc.Graph(figure=createscatter2d(solutions_plot_request))
            click_pause = 1
        else:
            navigator_graph_new = extend_navigator_plot(
                ranges_plot_request, navigator_graph
            )
            scatter_graph = dcc.Graph(figure=createscatter2d(solutions_plot_request))
            click_pause = None
        if navigator_graph is None:
            navigator_graph_new = create_navigator_plot(ranges_plot_request)
        return (navigator_graph_new, scatter_graph, click_pause)


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
    Output("callback_blackhole", "children"),
    [Input("submitbutton", "n_clicks")],
    [*[State(f"textbox{i+1}", "value") for i in range(2)]],
)
def submitevent(submitclick, *preference_values):
    if submitclick is None:
        raise PreventUpdate
    preference_request = session["preference"]
    pref_value = pd.DataFrame(
        [preference_values],
        columns=preference_request.content["dimensions_data"].columns,
    )
    preference_request.response = pref_value
    session["preference"] = preference_request
    return None


if __name__ == "__main__":
    app.run_server(debug=True)
