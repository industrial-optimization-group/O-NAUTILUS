from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as ex
import plotly.graph_objects as go

from dash.dependencies import Input, Output, State, ALL
from flask import session


from onautilus.onautilus import ONAUTILUS
from UI.app import app
from UI.pages.tools import create_navigator_plot, extend_navigator_plot
from onautilus.preferential_func_eval import preferential_func_eval as pfe

from desdeo_tools.interaction.request import RequestError


def layout():
    objective_names = session["objective_names"]
    max_multiplier = session["problem"]._max_multiplier
    return html.Div(
        children=[
            dbc.Row(
                dbc.Col(
                    html.H1("Navigation", id="header_navigator"),
                    className="row justify-content-center",
                )
            ),
            # Rows of Navigation elements (input boxes on left column, plots on right)
            dbc.Row(
                [
                    dbc.Col(
                        html.H3("Aspiration Levels"), width={"size": 2, "offset": 1}
                    ),
                    dbc.Col(html.H3("Navigator View"), width=8),
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
                                        html.P(
                                            (
                                                f"Enter a value for {y}"
                                                f" ({'Maximized' if max_m == -1 else 'Minimized'})"
                                            ),
                                            className="card-title",
                                        ),
                                        dcc.Input(
                                            id={"type": "pref_input", "name": y},
                                            placeholder=f"Enter a value for {y}",
                                            type="number",
                                        ),
                                    ]
                                ),
                                width={"size": 2, "offset": 1},
                            ),
                            dbc.Col(
                                dcc.Graph(
                                    id={"type": "nav_graph", "name": y},
                                    config={"displayModeBar": False},
                                ),
                                width=8,
                            ),
                        ],
                        className="h-25",
                    )
                    for y, max_m in zip(objective_names, max_multiplier)
                ],
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.ButtonGroup(
                            [
                                dbc.Button(
                                    "Pause",
                                    id="pausebutton",
                                    n_clicks=None,
                                    className="mr-1 mt-1 mb-3",
                                    color="primary",
                                ),
                                dbc.Button(
                                    "Submit",
                                    id="submitbutton",
                                    disabled=True,
                                    className="mr-1 mt-1 mb-3",
                                    color="primary",
                                ),
                                dbc.Button(
                                    "Extra Information",
                                    id="extrainfobutton",
                                    className="mr-1 mt-1 mb-3",
                                    color="primary",
                                ),
                            ]
                        ),
                        className="row justify-content-center",
                    )
                ]
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Collapse(
                        id="extra-info-collapse",
                        children=[
                            dbc.Card(
                                [
                                    html.H5(
                                        "Extra Information", className="card-title"
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                dbc.Col(
                                                    id="nav-graph-data", children=None
                                                )
                                            )
                                        ]
                                    ),
                                ]
                            ),
                            # dbc.Row(
                            #    dbc.Col([dcc.Graph(id="parallel-coordinates-plot")])
                            # ),
                        ],
                        is_open=False,
                    )
                )
            ),
            html.Div(id="callback_blackhole", hidden=True),
            dcc.Interval(
                id="step_counter", interval=5 * 100, n_intervals=0
            ),  # in milliseconds
            dbc.Row(dbc.Col(html.Div(id="pausewindow"))),
            html.Div(id="continue_iterate", n_clicks=0, children=1, hidden=True),
        ]
    )


def extra_info_card(
    name,
    is_minimized,
    preference,
    global_max,
    optimistic_reachable_max,
    known_reachable_max,
    iteration_point,
    known_reachable_min,
    optimistic_reachable_min,
    global_min,
):
    return dbc.Card(
        [
            html.H5(name, className="card-title"),
            dbc.CardBody(
                dbc.ListGroup(
                    [
                        dbc.ListGroupItem(f"Is minimized: {is_minimized}"),
                        dbc.ListGroupItem(f"Previous preference: {preference}"),
                        dbc.ListGroupItem(f"Global max: {global_max}"),
                        dbc.ListGroupItem(
                            f"Optimistic reachable range max: {optimistic_reachable_max}"
                        ),
                        dbc.ListGroupItem(
                            f"Known reachable range max: {known_reachable_max}"
                        ),
                        dbc.ListGroupItem(f"Iteration point: {iteration_point}"),
                        dbc.ListGroupItem(
                            f"Known reachable range min: {known_reachable_min}"
                        ),
                        dbc.ListGroupItem(
                            f"Optimistic reachable range min: {optimistic_reachable_min}"
                        ),
                        dbc.ListGroupItem(f"Global min: {global_min}"),
                    ]
                )
            ),
        ]
    )


@app.callback(
    [
        Output({"type": "nav_graph", "name": ALL}, "figure"),
        Output("continue_iterate", "children"),
        Output("nav-graph-data", "children"),
        # Output("parallel-coordinates-plot", "figure"),
    ],
    [Input(component_id="step_counter", component_property="n_intervals")],
    [
        State({"type": "nav_graph", "name": ALL}, "figure"),
        State("continue_iterate", "children"),
    ],
)
def iterate(n, nav_figures, continue_iterate_state):
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
            num_steps=100,
        )
        session["navigator"] = method
        ranges_plot_request, _, preference_request = method.requests()
        session["preference"] = preference_request
        session["preference_provided"] = False
        nav_figures = [
            create_navigator_plot(ranges_plot_request, objective_name, True, True)
            for objective_name in objective_names
        ]
        if "pref_value" not in session.keys():
            session["pref_value"] = pd.DataFrame(columns=objective_names, index=[0])
        continue_iterate = 0
    # Other cases
    else:
        if continue_iterate_state == 0:
            if not session["preference_provided"]:
                raise PreventUpdate
        objective_names = session["objective_names"]
        method = session["navigator"]
        preference_request = session["preference"]
        ranges_plot_request, _, preference_request = method.iterate(preference_request)
        session["preference"] = preference_request
        if preference_request.interaction_priority == "required":
            continue_iterate = 0
            session["preference_provided"] = False
        else:
            nav_figures = [
                extend_navigator_plot(ranges_plot_request, objective_name, nav_figure)
                for nav_figure, objective_name in zip(nav_figures, objective_names)
            ]
            continue_iterate = 1
        # if navigator_graphs is None:
        #     navigator_graph_new = create_navigator_plot(ranges_plot_request)
    # Extra information cards
    extra_info = []
    extra_data = method.request_solutions_plot()
    for y in objective_names:
        minimize = extra_data.content["dimensions_data"].loc["minimize"][y]
        global_max, global_min = [
            extra_data.content["dimensions_data"].loc["nadir"][y],
            extra_data.content["dimensions_data"].loc["ideal"][y],
        ][::minimize]
        optimistic_reachable_max, optimistic_reachable_min = [
            extra_data.content["dimensions_data"].loc["nadir_optimistic"][y],
            extra_data.content["dimensions_data"].loc["ideal_optimistic"][y],
        ][::minimize]
        known_reachable_max, known_reachable_min = [
            extra_data.content["dimensions_data"].loc["nadir_known"][y],
            extra_data.content["dimensions_data"].loc["ideal_known"][y],
        ][::minimize]
        iteration_point = extra_data.content["current_point"][y][0]
        preference = session["pref_value"][y][0]
        extra_info.append(
            extra_info_card(
                y,
                True if minimize == 1 else False,
                preference,
                global_max,
                optimistic_reachable_max,
                known_reachable_max,
                iteration_point,
                known_reachable_min,
                optimistic_reachable_min,
                global_min,
            )
        )
    extra_info = dbc.CardGroup(extra_info)

    # parallel coords
    """data = extra_data.content["data"]
    data["Type"] = "Unreachable"
    data["Type"][extra_data.content["achievable_ids"]] = "Known data"

    optimistic_data = extra_data.content["optimistic_data"]
    optimistic_data["Type"] = "Unreachable"
    optimistic_data["Type"][
        extra_data.content["achievable_ids_opt"]
    ] = "Optimistic data"

    ideal = extra_data.content["dimensions_data"].loc["ideal"]
    nadir = extra_data.content["dimensions_data"].loc["nadir"]
    ideal_known = extra_data.content["dimensions_data"].loc["ideal_known"]
    nadir_known = extra_data.content["dimensions_data"].loc["nadir_known"]
    ideal_optimistic = extra_data.content["dimensions_data"].loc["ideal_optimistic"]
    nadir_optimistic = extra_data.content["dimensions_data"].loc["nadir_optimistic"]
    ideal["Type"] = "ideal"
    nadir["Type"] = "nadir"
    ideal_known["Type"] = "ideal_known"
    nadir_known["Type"] = "nadir_known"
    ideal_optimistic["Type"] = "ideal_optimistic"
    nadir_optimistic["Type"] = "nadir_optimistic"
    data = pd.concat(
        [
            data,
            optimistic_data,
            ideal,
            nadir,
            ideal_known,
            nadir_known,
            ideal_optimistic,
            nadir_optimistic,
        ]
    )
    par_coord_fig = ex.parallel_coordinates(data)"""
    return (nav_figures, continue_iterate, extra_info)


@app.callback(
    [
        Output("step_counter", "disabled"),
        Output("pausebutton", "children"),
        Output("submitbutton", "disabled"),
        Output({"type": "pref_input", "name": ALL}, "disabled"),
        Output("pausewindow", "children"),
    ],
    [Input("pausebutton", "n_clicks")],
    [
        State("pausebutton", "children"),
        State({"type": "pref_input", "name": ALL}, "disabled"),
    ],
)
def pauseevent(pauseclick, pausevalue, input_box_state):
    if pauseclick is None:
        # Prevents pausing when initializing or other non-pausing events
        raise PreventUpdate
        # return (True, "Play", False, False, False)
    if pausevalue == "Pause":
        if pauseclick == 0:
            return (False, "Pause", True, [True] * len(input_box_state), None)
        return (True, "Play", False, [False] * len(input_box_state), pausewindow())
    # If paused
    elif pausevalue == "Play":
        if session["preference_provided"]:
            # Start playing if preference provided
            return (False, "Pause", True, [True] * len(input_box_state), None)
        else:
            raise PreventUpdate


@app.callback(
    Output("callback_blackhole", "children"),
    [Input("submitbutton", "n_clicks")],
    [State({"type": "pref_input", "name": ALL}, "value")],
)
def submitevent(submitclick, preference_values):
    if submitclick is None:
        raise PreventUpdate
    preference_request = session["preference"]
    pref_value = pd.DataFrame(
        [preference_values],
        columns=preference_request.content["dimensions_data"].columns,
    )
    session["pref_value"] = pref_value
    try:
        preference_request.response = pref_value
        session["preference_provided"] = True
    except RequestError as err:
        error_statement = err
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
                dbc.Row(
                    dbc.Col(
                        html.H3("New function evaluations"),
                        className="row justify-content-center",
                    )
                ),
                dbc.Row(
                    dbc.Col(
                        html.H5(
                            children="Enter a reference point for m-ASF calculation"
                        ),
                        className="row justify-content-center",
                    )
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        html.P(
                                            (f"Enter a value for {y}"),
                                            className="card-title",
                                        ),
                                        dcc.Input(
                                            id={"type": "mei-preference", "index": i},
                                            placeholder=f"Enter a value for {y}",
                                            type="number",
                                            value=reference[i],
                                        ),
                                    ]
                                )
                            ],
                            width=2,
                        )
                        for i, y in enumerate(objective_names)
                    ],
                    justify="center",
                ),
                dbc.Row(
                    dbc.Col(
                        dbc.Button(
                            "Evaluate new point",
                            id="functionevaluation",
                            className="mr-1 mt-1",
                            color="primary",
                        ),
                        className="row justify-content-center",
                    )
                ),
                dbc.Row(
                    dbc.Col(
                        dcc.Graph(id="func_eval_results", figure=scatter_graph),
                        className="row justify-content-center",
                    )
                ),
                dbc.Row(
                    dbc.Col(
                        dcc.Link("Go back to training page", href="/train#O-NAUTILUS"),
                        className="row justify-content-center",
                    )
                ),
            ]
        ),
    )


@app.callback(
    Output("func_eval_results", "figure"),
    [Input("functionevaluation", "n_clicks")],
    [
        State({"type": "mei-preference", "index": ALL}, "value"),
        State("func_eval_results", "figure"),
    ],
)
def function_evaluation(button_press, mei_prefs, updated_scatter_graph):
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
    ref_point = np.asarray(mei_prefs) * problem._max_multiplier
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


@app.callback(
    Output("extra-info-collapse", "is_open"),
    [Input("extrainfobutton", "n_clicks")],
    [State("extra-info-collapse", "is_open")],
)
def extra_info_collapse(pressed, state):
    if pressed is None:
        raise PreventUpdate
    return not state


if __name__ == "__main__":
    app.run_server(debug=True)
