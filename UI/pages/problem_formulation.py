from flask import session

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from optproblems.zdt import ZDT1, ZDT2, ZDT3
from dash.exceptions import PreventUpdate
import numpy as np

from UI.app import app


def zdt1func(x):
    x = np.asarray(x)
    evaluate = ZDT1()
    if x.ndim == 2:
        return np.asarray([evaluate(xi) for xi in x])
    elif x.ndim == 1:
        return evaluate(x)


def zdt2func(x):
    x = np.asarray(x)
    evaluate = ZDT2()
    if x.ndim == 2:
        return np.asarray([evaluate(xi) for xi in x])
    elif x.ndim == 1:
        return evaluate(x)


def zdt3func(x):
    x = np.asarray(x)
    evaluate = ZDT3()
    if x.ndim == 2:
        return np.asarray([evaluate(xi) for xi in x])
    elif x.ndim == 1:
        return evaluate(x)


test_problems = {"ZDT1": zdt1func, "ZDT2": zdt2func, "ZDT3": zdt3func}


def layout():
    names = session["original_dataset"].columns.tolist()
    return html.Div(
        [
            html.H1("Problem Formulation", id="header_problem_formulation"),
            # Hidden div storing column names
            html.Div(id="column_names", style={"display": "none"}, children=names),
            # Dropdown 1: decision variables
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
            # Dropdown 2: objective function names
            html.Label(
                [
                    "Choose objectives",
                    dcc.Dropdown(
                        id="objectives",
                        options=[
                            {"label": name, "value": name, "disabled": False}
                            for name in names
                        ],
                        value=[name for name in names if "y" in name or "f" in name],
                        placeholder="Choose objectives",
                        multi=True,
                    ),
                ]
            ),
            html.Label(
                [
                    "Provide analytical functions",
                    html.Div(id="analytical_function_inputs", children=[]),
                ]
            ),
            html.Label(
                [
                    "Choose test functions:",
                    dcc.Dropdown(
                        id="test_functions",
                        options=[
                            {"label": name, "value": name} for name in test_problems
                        ],
                        placeholder="Choose a test function",
                        multi=False,
                    ),
                ]
            ),
            html.Div(id="callback_blackhole_train", hidden=True),
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
    [
        Output("decision_variables", "options"),
        Output("analytical_function_inputs", "children"),
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

    return (decision_variable_names_restricted_options, analytical_function_inputs)


@app.callback(
    Output("callback_blackhole_train", "children"),
    [Input("test_functions", "value")],
)
def save_test_function(chosen_func):
    if chosen_func is None:
        raise PreventUpdate
    session["true_function"] = test_problems[chosen_func]
    return
