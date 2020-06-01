from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as ex
import plotly.graph_objects as go

from dash.dependencies import Input, Output, State
from flask import session


from onautilus.onautilus import ONAUTILUS
from UI.app import app
from UI.pages.tools2 import create_navigator_plot, extend_navigator_plot
from onautilus.preferential_func_eval import preferential_func_eval as pfe


def layout():
    objective_names = session["objective_names"]
    max_multiplier = session["problem"]._max_multiplier
    return html.Div(
        children=[
            dbc.Row(html.H1("Navigation", id="header_navigator")),
            # Rows of Navigation elements (input boxes on left column, plots on right)
            dbc.Row(
                [
                    dbc.Col(html.H3("Aspiration Levels"), width=2),
                    dbc.Col(html.H3("Navigator View"), width=9),
                ]
            ),
            html.Div(
                id="navigation-elements",
                children=[
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(
                                    [
                                        html.H4(
                                            (
                                                f"Enter a value for {y}"
                                                f" ({'Maximized' if max_m == -1 else 'Minimized'})"
                                            ),
                                            className="card-title",
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
                            dbc.Col(
                                dcc.Graph(
                                    id=f"nav_graph{i+1}",
                                    config={"displayModeBar": False},
                                ),
                                width=9,
                            ),
                        ],
                        className="h-25",
                    )
                    for i, (y, max_m) in enumerate(zip(objective_names, max_multiplier))
                ],
            ),
            html.Button("Pause", id="pausebutton", n_clicks=None),
            html.Button("Submit", id="submitbutton", disabled=True),
            dcc.Interval(
                id="step_counter", interval=1 * 1000, n_intervals=0
            ),  # in milliseconds
            html.Div(id="pausewindow"),
            html.Div(id="callback_blackhole"),
        ]
    )


@app.callback(
    [
        Output(component_id="navigation-elements", component_property="children"),
        Output("pausebutton", "n_clicks"),
    ],
    [Input(component_id="step_counter", component_property="n_intervals")],
    [State(component_id="navigation-elements", component_property="children")],
)
def iterate(n, navigation_elements):
    if n is None:
        raise PreventUpdate
    # Zeroth iteration
    if n == 0:
        objective_names = session["objective_names"]
        max_multiplier = session["problem"]._max_multiplier
        known_data = session["original_dataset"][objective_names].values
        surrogate_data = session["optimistic_data"][objective_names].values
        method = ONAUTILUS(
            known_data=known_data,
            optimistic_data=surrogate_data,
            objective_names=objective_names,
            max_multiplier=max_multiplier,
            num_steps=20,
        )
        session["navigator"] = method
        ranges_plot_request, _, preference_request = method.requests()
        session["preference"] = preference_request

        for i, objective_name in enumerate(objective_names):
            navigator_graph = create_navigator_plot(
                ranges_plot_request, objective_name, True, True
            )
            navigation_elements[i]["props"]["children"][1]["props"]["children"][
                "props"
            ]["figure"] = navigator_graph
        return (navigation_elements, 1)
    # Other cases
    else:
        objective_names = session["objective_names"]
        method = session["navigator"]
        preference_request = session["preference"]
        ranges_plot_request, _, preference_request = method.iterate(preference_request)
        session["preference"] = preference_request
        if preference_request.interaction_priority == "required":
            click_pause = 1
        else:
            for i, objective_name in enumerate(objective_names):
                navigator_graph = navigation_elements[i]["props"]["children"][1][
                    "props"
                ]["children"]["props"]["figure"]
                navigator_graph = extend_navigator_plot(
                    ranges_plot_request, objective_name, navigator_graph
                )
                navigation_elements[i]["props"]["children"][1]["props"]["children"][
                    "props"
                ]["figure"] = navigator_graph
            click_pause = 0
        # if navigator_graphs is None:
        #     navigator_graph_new = create_navigator_plot(ranges_plot_request)
        return (navigation_elements, click_pause)


@app.callback(
    [
        Output("step_counter", "disabled"),
        Output("pausebutton", "children"),
        Output("submitbutton", "disabled"),
        Output("pausewindow", "children"),
    ],
    [Input("pausebutton", "n_clicks")],
    [State("pausebutton", "children")],
)
def pauseevent(pauseclick, pausevalue):
    if pauseclick is None:
        # Prevents pausing when initializing or other non-pausing events
        return (True, "Play", False, pausewindow())
        # return (True, "Play", False, False, False)
    if pausevalue == "Pause":
        if pauseclick == 0:
            return (False, "Pause", True, None)
        return (True, "Play", False, pausewindow())
    elif pausevalue == "Play":
        return (False, "Pause", True, None)


@app.callback(
    Output("callback_blackhole", "children"),
    [Input("submitbutton", "n_clicks")],
    [State("navigation-elements", "children")],
)
def submitevent(submitclick, navigation_elements):
    if submitclick is None:
        raise PreventUpdate
    preference_request = session["preference"]
    preference_values = [
        element["props"]["children"][0]["props"]["children"]["props"]["children"][1][
            "props"
        ]["value"]
        for element in navigation_elements
    ]
    pref_value = pd.DataFrame(
        [preference_values],
        columns=preference_request.content["dimensions_data"].columns,
    )
    session["pref_value"] = pref_value
    preference_request.response = pref_value
    session["preference"] = preference_request
    return None


def pausewindow():
    long_data = session["navigator"].request_long_data()
    objective_names = session["objective_names"]
    reference = long_data[long_data["Source"] == "Preference point"][
        objective_names
    ].values[0]
    scatter_graph = ex.scatter_matrix(
        long_data, dimensions=session["objective_names"], color="Source"
    )
    return (
        html.Div(
            [
                html.Label(
                    [
                        "New function evaluations",
                        html.H5(children="Enter a reference point for MEI calculation"),
                        html.Div(
                            id="mei_text_boxes",
                            children=[
                                dcc.Input(
                                    id=f"meibox{i+1}",
                                    placeholder=f"Enter a value for {y}",
                                    type="number",
                                    value=reference[i],
                                )
                                for i, y in enumerate(objective_names)
                            ],
                        ),
                        html.Button("Evaluate new point", id="functionevaluation"),
                        html.Div(
                            id="func_eval_results_div",
                            children=[
                                dcc.Graph(id="func_eval_results", figure=scatter_graph)
                            ],
                        ),
                        dcc.Link("Go back to training page", href="/train#O-NAUTILUS"),
                    ]
                )
            ]
        ),
    )


@app.callback(
    Output("func_eval_results", "figure"),
    [Input("functionevaluation", "n_clicks")],
    [State("mei_text_boxes", "children"), State("func_eval_results", "figure")],
)
def function_evaluation(button_press, mei_div, updated_scatter_graph):
    if button_press is None:
        raise PreventUpdate
    if updated_scatter_graph is not None:
        scatter_graph = updated_scatter_graph

    ranges_request = session["navigator"].request_ranges_plot()
    optimizer = session["optimizer"]
    true_func_eval = session["true_function"]
    data = session["original_dataset"]
    optimistic_data = session["optimistic_data"]
    problem = optimizer.population.problem
    ref_point = (
        np.asarray([element["props"]["value"] for element in mei_div])
        * problem._max_multiplier
    )
    var_names = problem.get_variable_names()
    obj_names = problem.get_objective_names()
    ideal_current = (
        ranges_request.content["dimensions_data"].loc["ideal"].values
        * problem._max_multiplier
    )
    nadir_current = (
        ranges_request.content["dimensions_data"].loc["nadir"].values
        * problem._max_multiplier
    )

    x_new = np.atleast_2d(
        pfe(
            problem,
            optimistic_data,
            ideal_current,
            nadir_current,
            reference_point=ref_point,
        ).result.xbest
    )
    y_new = true_func_eval(x_new)

    y_new_predicted = problem.evaluate(x_new, use_surrogate=True).objectives

    x = data[var_names].values
    y = data[obj_names].values
    x = np.vstack((x, x_new))
    y = np.vstack((y, y_new))
    data = np.hstack((x, y))
    data = pd.DataFrame(data, columns=var_names + obj_names)
    session["original_dataset"] = data
    ref_point = ref_point * problem._max_multiplier
    ref_point_fig = go.Splom(
        name="Reference Point",
        dimensions=[
            dict(label=obj, values=[ref_point[i]]) for i, obj in enumerate(obj_names)
        ],
    )
    before_pred_fig = go.Splom(
        name="Prediction",
        dimensions=[
            dict(label=obj, values=y_new_predicted[:, i])
            for i, obj in enumerate(obj_names)
        ],
    )
    after_pred_fig = go.Splom(
        name="After evaluation",
        dimensions=[
            dict(label=obj, values=y_new[:, i]) for i, obj in enumerate(obj_names)
        ],
    )
    return go.Figure(
        scatter_graph["data"] + [ref_point_fig, before_pred_fig, after_pred_fig]
    )


if __name__ == "__main__":
    app.run_server(debug=True)
