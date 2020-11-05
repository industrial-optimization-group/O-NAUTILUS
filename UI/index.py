import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
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
        dbc.Row(
            dbc.Col(
                html.H1("Choose an application"), className="row justify-content-center"
            )
        ),
        dbc.Row(
            dbc.Col(
                dbc.ButtonGroup(
                    [
                        dbc.Button(
                            "O-NAUTILUS",
                            id="onautilus_button",
                            n_clicks_timestamp=-1,
                            href="/upload#O-NAUTILUS",
                            className="mr-1 mt-1",
                            color="primary",
                        ),
                        dbc.Button(
                            "Coloured Parallel Coordinates",
                            id="cpc_button",
                            n_clicks_timestamp=-1,
                            href="/upload#CPC",
                            className="mr-1 mt-1",
                            color="primary",
                        ),
                    ]
                ),
                className="row justify-content-center",
            )
        ),
    ]
)


def navbuttons(prev, home, next_):
    buttons = [
        dbc.Row(
            dbc.Col(
                dbc.ButtonGroup(
                    [
                        dbc.Button(
                            "Previous",
                            id="prev_button",
                            n_clicks_timestamp=-1,
                            href=prev,
                            className="mt-3",
                            color="primary",
                        ),
                        dbc.Button(
                            "Home",
                            id="home_button",
                            n_clicks_timestamp=-1,
                            href=home,
                            className="ml-1 mr-1 mt-3",
                            color="primary",
                        ),
                        dbc.Button(
                            "Next",
                            id="next_button",
                            n_clicks_timestamp=-1,
                            href=next_,
                            className="mt-3",
                            color="primary",
                        ),
                    ]
                ),
                className="row justify-content-center",
            )
        )
    ]
    return buttons


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
        prev = "/"
        next_ = "/error"
        if app_choice == "#O-NAUTILUS":
            current_page_index = o_nautilus_page_order.index(pathname)
            if current_page_index != 0:
                prev = o_nautilus_page_order[current_page_index - 1]
            if current_page_index != len(o_nautilus_page_order) - 1:
                next_ = o_nautilus_page_order[current_page_index + 1]

        elif app_choice == "#CPC":
            current_page_index = parallel_coords_colouring_order.index(pathname)
            if current_page_index != 0:
                prev = parallel_coords_colouring_order[current_page_index - 1]

            if current_page_index != len(parallel_coords_colouring_order) - 1:
                next_ = parallel_coords_colouring_order[current_page_index + 1]
        else:
            return "404"

        layout.children.extend(
            deepcopy(
                navbuttons(prev + app_choice, "/" + app_choice, next_ + app_choice)
            )
        )
        return layout
    else:
        return "404"


if __name__ == "__main__":
    app.run_server(debug=False)
