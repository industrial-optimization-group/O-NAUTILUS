from flask import session

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from onautilus.problems import (
    zdt1func,
    zdt2func,
    zdt3func,
    riverfunc,
    vehicle_crash_worthiness,
    rocket_injector,
)
from dash.exceptions import PreventUpdate
import numpy as np
import pandas as pd

from UI.app import app


test_problems = {
    "ZDT1": zdt1func,
    "ZDT2": zdt2func,
    "ZDT3": zdt3func,
    "River pollution problem": riverfunc,
    "Vehicle crashworthiness design problem": vehicle_crash_worthiness,
    "Rocket injector design problem": rocket_injector,
}


def layout():
    names = session["original_dataset"].columns.tolist()
    return html.Div(
        [
            dbc.Row(
                dbc.Col(
                    html.H1("Problem Formulation", id="header_problem_formulation"),
                    className="row justify-content-center",
                )
            ),
            # Hidden div storing column names
            html.Div(id="column_names", style={"display": "none"}, children=names),
            # Dropdown 1: decision variables
            dbc.Row(
                dbc.Col(
                    html.Label(
                        id="testlabel",
                        children=[
                            "Choose decision variables",
                            dcc.Dropdown(
                                id="decision_variables",
                                options=[
                                    {"label": name, "value": name, "disabled": False}
                                    for name in names
                                ],
                                value=[name for name in names if "x" in name],
                                placeholder="Choose decision variables",
                                multi=True,
                            ),
                        ],
                    ),
                    className="row justify-content-center",
                )
            ),
            # Radio button 1: Decision variables
            dbc.Row(
                dbc.Col(
                    html.Label(
                        id="bounds",
                        children=[
                            "Choose lower and upper bounds:",
                            dcc.RadioItems(
                                id="bounds_button",
                                options=[
                                    {"label": "0-1 for all variables", "value": "0-1"},
                                    {
                                        "label": "Min-Max from dataset",
                                        "value": "min-max",
                                    },
                                ],
                            ),
                        ],
                    ),
                    className="row justify-content-center",
                )
            ),
            # Dropdown 2: objective function names
            dbc.Row(
                dbc.Col(
                    html.Label(
                        [
                            "Choose objectives",
                            dcc.Dropdown(
                                id="objectives",
                                options=[
                                    {"label": name, "value": name, "disabled": False}
                                    for name in names
                                ],
                                value=[
                                    name for name in names if "y" in name or "f" in name
                                ],
                                placeholder="Choose objectives",
                                multi=True,
                                style={"width": "300px"},
                            ),
                        ]
                    ),
                    className="row justify-content-center",
                )
            ),
            # Dropdown: maximize
            dbc.Row(
                dbc.Col(
                    html.Label(
                        [
                            "Choose objectives to be maximized",
                            dcc.Dropdown(
                                id="objectives_max_info",
                                options=[
                                    {"label": name, "value": name, "disabled": False}
                                    for name in names
                                ],
                                value=[
                                    name for name in names if "y" in name or "f" in name
                                ],
                                placeholder="Choose objectives to be maximized",
                                multi=True,
                                style={"width": "300px"},
                            ),
                        ]
                    ),
                    className="row justify-content-center",
                )
            ),
            # Dropdown: choose a test function
            dbc.Row(
                dbc.Col(
                    html.Label(
                        [
                            "Choose test functions:",
                            dcc.Dropdown(
                                id="test_functions",
                                options=[
                                    {"label": name, "value": name}
                                    for name in test_problems
                                ],
                                placeholder="Choose a test function",
                                multi=False,
                                style={"width": "300px"},
                            ),
                        ]
                    ),
                    className="row justify-content-center",
                )
            ),
            html.Label(["Problem Information:", html.Div(id="prob_info", children=[])]),
            html.Div(id="callback_blackhole_train", hidden=True),
            html.Div(id="callback_blackhole2_train", hidden=True),
        ]
    )


@app.callback(
    Output("objectives", "options"),
    [Input("decision_variables", "value")],
    [State("column_names", "children")],
)
def add_decision_vars(decision_variable_names, all_names):
    if decision_variable_names is None:
        raise PreventUpdate
    session["decision_variable_names"] = decision_variable_names
    objective_names_restricted_options = [
        {
            "label": name,
            "value": name,
            "disabled": True if name in decision_variable_names else False,
        }
        for name in all_names
    ]
    return objective_names_restricted_options


@app.callback(
    [Output("decision_variables", "options"), Output("objectives_max_info", "options")],
    [Input("objectives", "value")],
    [State("column_names", "children")],
)
def add_objectives(objective_names, all_names):
    if objective_names is None:
        raise PreventUpdate
    session["objective_names"] = objective_names
    decision_variable_names_restricted_options = [
        {
            "label": name,
            "value": name,
            "disabled": True if name in objective_names else False,
        }
        for name in all_names
    ]
    max_info_options = [{"label": name, "value": name} for name in objective_names]

    return (decision_variable_names_restricted_options, max_info_options)


@app.callback(
    Output("callback_blackhole_train", "children"),
    [Input("bounds_button", "value")],
    [State("decision_variables", "value")],
)
def bounds(bound_type, decision_variables):
    if bound_type is None:
        raise PreventUpdate
    if decision_variables is None:
        raise PreventUpdate
    if bound_type == "0-1":
        lower_bounds = [0] * len(decision_variables)
        upper_bounds = [1] * len(decision_variables)
        bounds = pd.DataFrame(
            np.vstack((lower_bounds, upper_bounds)),
            columns=decision_variables,
            index=["lower_bound", "upper_bound"],
        )
    if bound_type == "min-max":
        bounds = (
            session["original_dataset"]
            .describe()[decision_variables]
            .loc[["min", "max"]]
        )
        bounds.rename(index={"min": "lower_bound", "max": "upper_bound"}, inplace=True)
    session["bounds"] = bounds
    return


@app.callback(
    Output("callback_blackhole2_train", "children"), [Input("test_functions", "value")]
)
def save_test_function(chosen_func):
    if chosen_func is None:
        raise PreventUpdate
    session["true_function"] = test_problems[chosen_func]
    return


@app.callback(
    Output("prob_info", "children"),
    [Input("objectives_max_info", "value")],
    [
        State("test_functions", "value"),
        State("decision_variables", "value"),
        State("objectives", "value"),
    ],
)
def maximization_info(max_obj, prob_name, var_name, obj_name):
    max_data = pd.DataFrame(columns=obj_name, index=[0])
    max_data[:] = False
    max_data[max_obj] = True
    session["maximization_info"] = max_data
    prob_info = (
        f"Problem name is {prob_name}.\n"
        f"Decision variables are {var_name}.\n"
        f"Objective variables are {obj_name}.\n"
        f"Objectives to be maximized are {max_obj}."
    )
    return prob_info


"""# Textbox: Provide analytical functions
dbc.Row(
    dbc.Col(
        html.Label(
            [
                "Provide analytical functions",
                html.Div(id="analytical_function_inputs", children=[]),
            ]
        )
    )
),

@app.callback(
    [
        Output("decision_variables", "options"),
        Output("analytical_function_inputs", "children"),
        Output("objectives_max_info", "options"),
    ],
    [Input("objectives", "value")],
    [State("column_names", "children")],
)
def add_objectives(objective_names, all_names):
    if objective_names is None:
        raise PreventUpdate
    session["objective_names"] = objective_names
    decision_variable_names_restricted_options = [
        {
            "label": name,
            "value": name,
            "disabled": True if name in objective_names else False,
        }
        for name in all_names
    ]
    max_info_options = [{"label": name, "value": name} for name in objective_names]
    analytical_function_inputs = [
        html.Label(
            [
                f"Enter the analytical function for {objective}",
                dcc.Input(
                    placeholder=f"Enter the analytical function for {objective}",
                    type="text",
                    value="",
                ),
            ]
        )
        for objective in objective_names
    ]
    return (
        decision_variable_names_restricted_options,
        analytical_function_inputs,
        max_info_options,
    )

"""
