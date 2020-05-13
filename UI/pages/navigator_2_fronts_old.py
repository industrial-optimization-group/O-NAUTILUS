from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as ex

from dash.dependencies import Input, Output, State
from flask import session


from onautilus.onautilus import ONAUTILUS
from UI.app import app
from UI.pages.tools import create_navigator_plot, extend_navigator_plot, createscatter2d
from onautilus.preferential_func_eval import preferential_func_eval as pfe


def layout():
    objective_names = session["objective_names"]
    return html.Div(
        children=[
            html.H1("Navigation", id="header_navigator"),
            html.Div(
                id="navigator_graph",
                style={"width": "49%", "display": "inline-block"},
                children=html.Label(
                    ["Navigator View", dcc.Graph(id="navigator_graph_figure")]
                ),
            ),
            html.Div(
                id="scatter_graph_div",
                style={"width": "49%", "display": "inline-block"},
                children=html.Label(
                    ["Conventional View", dcc.Graph(id="scatter_graph")]
                ),
            ),
            html.Label(
                [
                    "Provide aspiration levels",
                    html.Div(
                        id="aspiration_text_boxes",
                        children=[
                            dcc.Input(
                                id=f"textbox{i+1}",
                                placeholder=f"Enter a value for {y}",
                                type="number",
                            )
                            for i, y in enumerate(objective_names)
                        ],
                        hidden=False,
                    ),
                ]
            ),
            html.Button("Pause", id="pausebutton"),
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
        Output(component_id="navigator_graph_figure", component_property="figure"),
        Output("scatter_graph", "figure"),
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
        ranges_plot_request, solutions_plot_request, preference_request = (
            method.requests()
        )
        if "preference" in session.keys():
            print(session["preference"])
            preference_request.response = session["preference"].response
        session["preference"] = preference_request
        navigator_graph_new = create_navigator_plot(ranges_plot_request)
        # scatter_graph = createscatter2d(solutions_plot_request)
        scatter_graph = []
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
            # scatter_graph = createscatter2d(solutions_plot_request)
            scatter_graph = []
            click_pause = 1
        else:
            navigator_graph_new = extend_navigator_plot(
                ranges_plot_request, navigator_graph
            )
            # scatter_graph = createscatter2d(solutions_plot_request)
            scatter_graph = None
            click_pause = 0
        if navigator_graph is None:
            navigator_graph_new = create_navigator_plot(ranges_plot_request)
        return (navigator_graph_new, scatter_graph, click_pause)


@app.callback(
    [
        Output("step_counter", "disabled"),
        Output("pausebutton", "children"),
        Output("submitbutton", "disabled"),
        Output("aspiration_text_boxes", "hidden"),
        Output("pausewindow", "children")
        # *[Output(f"textbox{i+1}", "value") for i in range(2)]
    ],
    [Input("pausebutton", "n_clicks")],
    [State("pausebutton", "children"), State("scatter_graph", "figure")],
)
def pauseevent(pauseclick, pausevalue, scatter_graph):
    if pauseclick is None:
        # Prevents pausing when initializing or other non-pausing events
        return (True, "Play", False, False, pausewindow(scatter_graph))
        # return (True, "Play", False, False, False)
    if pausevalue == "Pause":
        if pauseclick == 0:
            return (False, "Pause", True, True, None)
        return (True, "Play", False, False, pausewindow(scatter_graph))
    elif pausevalue == "Play":
        return (False, "Pause", True, True, None)


@app.callback(
    Output("callback_blackhole", "children"),
    [Input("submitbutton", "n_clicks")],
    [State("aspiration_text_boxes", "children")],
)
def submitevent(submitclick, preference_value_div):
    if submitclick is None:
        raise PreventUpdate
    preference_request = session["preference"]
    preference_values = [element["props"]["value"] for element in preference_value_div]
    pref_value = pd.DataFrame(
        [preference_values],
        columns=preference_request.content["dimensions_data"].columns,
    )
    preference_request.response = pref_value
    session["preference"] = preference_request
    return None


def pausewindow(scatter_graph):
    ranges_request = session["navigator"].request_ranges_plot()
    reference = ranges_request.content["current_point"].values[0]
    objective_names = session["objective_names"]
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
                            style={"width": "49%", "display": "inline-block"},
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
    [
        State("mei_text_boxes", "children"),
        State("scatter_graph", "children"),
        State("func_eval_results", "figure"),
    ],
)
def function_evaluation(
    button_press, mei_div, scatter_graph: ex.scatter, updated_scatter_graph
):
    if button_press is None:
        raise PreventUpdate
    if updated_scatter_graph is not None:
        scatter_graph = updated_scatter_graph
    ref_point = np.asarray([element["props"]["value"] for element in mei_div])

    ranges_request = session["navigator"].request_ranges_plot()
    optimizer = session["optimizer"]
    true_func_eval = session["true_function"]
    data = session["original_dataset"]

    optimistic_data = session["optimistic_data"]
    problem = optimizer.population.problem
    var_names = problem.get_variable_names()
    obj_names = problem.get_objective_names()
    ideal = ranges_request.content["dimensions_data"].loc["ideal"].values
    nadir = ranges_request.content["dimensions_data"].loc["nadir"].values

    x_new = pfe(
        problem, optimistic_data, ideal, nadir, reference_point=ref_point
    ).result.xbest
    y_new = true_func_eval(x_new)
    y_new_predicted = problem.evaluate(x_new, use_surrogate=True)

    x = data[var_names].values
    y = data[obj_names].values
    x = np.vstack((x, x_new))
    y = np.vstack((y, y_new))
    data = np.hstack((x, y))
    data = pd.DataFrame(data, columns=var_names + obj_names)
    session["original_dataset"] = data
    """print(scatter_graph)
    scatter_graph["data"] += [
        dict(
            x=y_new_predicted.objectives[:, 0],
            y=y_new_predicted.objectives[:, 1],
            error_x=dict(
                type="data", array=y_new_predicted.uncertainity[:, 0], visible=True
            ),
            error_y=dict(
                type="data", array=y_new_predicted.uncertainity[:, 1], visible=True
            ),
            name="Predicted result",
        )
    ]
    scatter_graph["data"] += [dict(x=[y_new[0]], y=[y_new[1]], name="Evaluated Result")]"""
    return scatter_graph


if __name__ == "__main__":
    app.run_server(debug=True)
