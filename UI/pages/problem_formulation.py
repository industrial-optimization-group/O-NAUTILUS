from flask import session

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from UI.app import app


def layout():
    names = session["original_dataset"].columns
    return html.Div(
        [
            html.H1("Problem Formulation", id="header_problem_formulation"),
            # Hidden div storing column names
            html.Div(id="column_names", style={"display": "none"}, children=names),
            # Dropdown 1: decision variables
            html.Label(
                [
                    "Choose decision variables",
                    dcc.Dropdown(
                        id="decision_variables",
                        options=[
                            {"label": name, "value": name, "disabled": False}
                            for name in names
                        ],
                        placeholder="Choose decision variables",
                        multi=True,
                    ),
                ]
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
                        placeholder="Choose objectives",
                        multi=True,
                    ),
                ]
            ),
        ]
    )


@app.callback(
    Output("objectives", "options"),
    [Input("decision_variables", "value")],
    [State("column_names", "children")],
)
def update_objectives(decision_variable_names, all_names):
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
    Output("decision_variables", "options"),
    [Input("objectives", "value")],
    [State("column_names", "children")],
)
def update_decision_variables(objective_names, all_names):
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
    return decision_variable_names_restricted_options