import base64
import datetime
import io
from flask_session import Session
from flask import session

import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import pandas as pd

app = dash.Dash(__name__)
app.server.secret_key = '_5#y2L"F4Q8z]/'
app.server.config.from_object("config.Config")
app.server.config["SESSION_TYPE"] = "filesystem"


sess = Session()
sess.init_app(app.server)


app.layout = html.Div(
    [
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            # Allow multiple files to be uploaded
            multiple=True,
        ),
        html.Div(id="output-data-upload"),
        html.Button("Previous", id="prev_data_upload"),
        html.Button("Home", id="home_data_upload"),
        html.Button("Next", id="next_data_upload"),
    ]
)


def parse_contents(contents, filename, date):
    """Parse and return csv/xls files as pandas DataFrame and Dash DataTable

    Parameters
    ----------
    contents : str
        Contents from the dcc.Upload component.
    filename : str
        Name of the uploaded file
    date : [type]
        Date and time of file upload

    Returns
    -------
    Tuple[pd.DataFrame, html.Div]
        The uploaded data in pandas DataFrame and dash datatable formats.
    """
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif "xls" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])

    return df


def create_datatable(df, filename):
    layout = html.Div(
        [
            html.H5(filename),
            dash_table.DataTable(
                data=df.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df.columns],
            ),
        ]
    )
    return layout


@app.callback(
    Output("output-data-upload", "children"),
    [Input("upload-data", "contents")],
    [State("upload-data", "filename"), State("upload-data", "last_modified")],
)
def update_layout(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        data_df = [
            parse_contents(c, n, d)
            for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)
        ][0]
        session["original_dataset"] = data_df
        layout = create_datatable(data_df, list_of_names[0])
        return layout


if __name__ == "__main__":
    app.run_server(debug=True)
