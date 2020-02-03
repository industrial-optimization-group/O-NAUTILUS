import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from copy import deepcopy


from UI.app import app
from UI.pages import (
    data_upload,
    problem_formulation,
    train_model,
    optimize,
    navigator_2_fronts,
    coloring_parallel_coords,
)

pages = {
    "/upload": data_upload,
    "/problem": problem_formulation,
    "/train": train_model,
    "/optimize": optimize,
    "/navigate": navigator_2_fronts,
    "/colour": coloring_parallel_coords,
}

o_nautilus_page_order = ["/upload", "/problem", "/train", "/optimize", "/navigate"]
parallel_coords_colouring_order = ["/upload", "/problem", "/colour"]


layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content", children=[]),
        html.Div(id="app_choice", children=None, hidden=True),
    ]
)

home_page = html.Div(
    [
        html.H1("Choose an application"),
        dcc.Link(
            html.Button("O-NAUTILUS", id="onautilus_button", n_clicks_timestamp=-1),
            href="/upload#O-NAUTILUS",
        ),
        dcc.Link(
            html.Button(
                "Coloured Parallel Coordinates", id="cpc_button", n_clicks_timestamp=-1
            ),
            href="/upload#CPC",
        ),
    ]
)

buttons = [
    dcc.Link(
        html.Button("Previous", id="prev_button", n_clicks_timestamp=-1), href="/"
    ),
    dcc.Link(html.Button("Home", id="home_button", n_clicks_timestamp=-1), href="/"),
    dcc.Link(
        html.Button("Next", id="next_button", n_clicks_timestamp=-1),
        href=o_nautilus_page_order[0],
    ),
]


app.layout = layout


@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")],
    [State("url", "hash")],
)
def display_page(pathname, app_choice):
    if pathname == "/":
        return home_page
    elif pathname in pages:
        layout = pages[pathname].layout()
        layout.children.extend(deepcopy(buttons))
        if app_choice == "#O-NAUTILUS":
            current_page_index = o_nautilus_page_order.index(pathname)
            if current_page_index != 0:
                layout.children[-3].href = (
                    o_nautilus_page_order[current_page_index - 1] + "#O-NAUTILUS"
                )
            if current_page_index != len(o_nautilus_page_order) - 1:
                layout.children[-1].href = (
                    o_nautilus_page_order[current_page_index + 1] + "#O-NAUTILUS"
                )
        elif app_choice == "#CPC":
            current_page_index = parallel_coords_colouring_order.index(pathname)
            if current_page_index != 0:
                layout.children[-3].href = (
                    parallel_coords_colouring_order[current_page_index - 1] + "#CPC"
                )
            if current_page_index != len(parallel_coords_colouring_order) - 1:
                layout.children[-1].href = (
                    parallel_coords_colouring_order[current_page_index + 1] + "#CPC"
                )
        else:
            return "404"
        return layout
    else:
        return "404"


"""@app.callback(
    Output("app_choice", "children"),
    [
        Input("onautilus_button", "n_clicks_timestamp"),
        Input("cpc_button", "n_clicks_timestamp"),
    ],
    [State("app_choice", "children")],
)
def app_choice(onautilus_button_time, cpc_button_time, app_choice_state):
    if app_choice_state is None:
        raise PreventUpdate
    if onautilus_button_time > cpc_button_time:
        return "O-NAUTILUS"
    elif cpc_button_time > onautilus_button_time:
        return "CPC"
    raise PreventUpdate"""


"""@app.callback(
    Output("url", "pathname"),
    [
        Input("next_button", "n_clicks"),
        Input("prev_button", "n_clicks"),
        Input("home_button", "n_clicks"),
    ],
    [
        State("url", "pathname"),
        State("next_button", "n_clicks_timestamp"),
        State("prev_button", "n_clicks_timestamp"),
        State("home_button", "n_clicks_timestamp"),
    ],
)
def button_press(
    next_button,
    prev_button,
    home_button,
    pathname,
    next_button_time,
    prev_button_time,
    home_button_time,
):
    if next_button is None or prev_button is None or home_button is None:
        raise PreventUpdate
    if next_button_time > home_button_time and next_button_time > prev_button_time:
        return next_button_press(pathname)
    elif prev_button_time > home_button_time and prev_button_time > next_button_time:
        return prev_button_press(pathname)
    else:
        return home_button_press(pathname)


def next_button_press(pathname):
    if pathname == "/":
        return o_nautilus_page_order[0]
    else:
        current_page_index = o_nautilus_page_order.index(pathname)
        return o_nautilus_page_order[current_page_index + 1]


def prev_button_press(pathname):
    if pathname == "/":
        raise PreventUpdate
    else:
        current_page_index = o_nautilus_page_order.index(pathname)
        return o_nautilus_page_order[current_page_index - 1]


def home_button_press(clicked, pathname):
    return ""
"""

if __name__ == "__main__":
    app.run_server(debug=False)
