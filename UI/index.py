import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


from UI.app import app
from UI.pages import data_upload, problem_formulation, train_model, optimize


app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/upload/":
        return data_upload.layout()
    elif pathname == "/problem/":
        return problem_formulation.layout()
    elif pathname == "/train/":
        return train_model.layout()
    elif pathname == "/optimize/":
        return optimize.layout()
    else:
        return "404"


if __name__ == "__main__":
    app.run_server(debug=True)
