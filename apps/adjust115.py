import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash.dash import no_update

import os
from urllib.parse import quote as urlquote
import base64
from datetime import datetime as dt

from app import navbar as navbar
from app import app


def directories_adjust():

    files_115 = [a[:7] for a in os.listdir(
        './monthly_data/115/') if a != '.gitkeep']

    files_20 = [a[:7] for a in os.listdir(
        './monthly_data/20/') if a != '.gitkeep']

    directories_adjust = {a for a in files_115 if a in files_20}

    return([{'label': i, 'value': i} for i in directories_adjust])


column_style = {'height': '100%', 'flex-direction': 'column',
                'display': 'flex', 'align-items': 'center',
                'justify-content': 'center',
                'boxShadow': '3px 3px 20px -10px #eabf1a',
                'marginLeft': '10%', 'marginRight': '10%',
                'marginBottom': '1%', 'marginTop': '10%'}


layout = html.Div([
    dcc.Store(id='new-memory'),
    navbar,
    # html.Iframe(src="https://www.youtube.com/embed/1CSeY10zbqo",
    #             style={'width':560 , 'height': 315, 'frameborder': 0}),
    dbc.Row([

        dbc.Col([
            dbc.Row([
                dbc.Alert("Please Select a Period", color="warning"),
                dbc.Button('update', id='update_dropdown_adjust',
                           style={'display': 'none'}),
                dcc.Dropdown(id='month_selection_adjust',
                             style={'width': '30vw', 'textAlign': 'center',
                                    'margin': '0 auto', 'marginBottom': 10},
                             options=directories_adjust()),
                html.A('Download 115 Alarms', id="download_btn_115",
                       style={'background-color': 'white',
                              'color': 'black', 'padding': '5px',
                              'text-decoration': 'none',
                              'border': '1px solid black',
                              }),
                html.A('Download 20-25 alarms', id="download_btn_20",
                       style={'background-color': 'white',
                              'color': 'black', 'padding': '5px',
                              'text-decoration': 'none',
                              'border': '1px solid black',
                              })
            ],
                no_gutters=True,
                style={'flex-direction': 'column', 'align-items': 'center',
                       'justify-content': 'center'}),

        ], style=column_style),

        dbc.Col([
            dbc.Button('Submit/Upload',
                       id='new-submit-button', n_clicks=0),
            dbc.Tooltip(
                "Please upload files after adjustment, "
                "And return to recalculate Availability",
                target="new-submit-button"),

            dcc.Loading(
                dcc.Upload(
                    id="adjusted-upload-data",
                    children=html.Div(
                        ["Drag and drop or click"
                         "to select "
                         "a file to upload."]),
                    style={"width": "100%",
                           "height": "60px",
                           "lineHeight": "60px",
                           "borderWidth": "1px",
                           "borderStyle": "dashed",
                           "borderRadius": "5px",
                           "textAlign": "center",
                           "margin": "10px"},
                    multiple=True,
                    max_size=-1),
                color='#eabf1a'),
            dcc.Loading(html.P(id="new-file-list"), color='#eabf1a')],
            style=column_style),

    ],
        style={'height': '50%', 'display': 'flex', 'align-items': 'center',
               'justify-content': 'center'},
        no_gutters=True)
], style={'height': '100vh'})


def file_download(path, filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = f"/download/uploads/{urlquote(path)}/{urlquote(filename)}"
    return html.A(filename, href=location)


@app.callback(
    [Output('download_btn_115', 'href'),
     Output('download_btn_20', 'href')],
    [Input('month_selection_adjust', 'value')])
# @cache.memoize(timeout=60)
def callback_download_adjust(x):

    if x is None:
        return None, None

    location_115 = (f"/download/115/{x}-115-missing.xlsx")

    # location = f"/download/{urlquote('results')}/anaconda.exe"
    location_20 = f"/download/20/{x}-cut-missing.xlsx"

    return (location_115,
            location_20
            )


@app.callback(
    [Output("adjusted-upload-data", "filename"),
     Output("adjusted-upload-data", "contents"),
     Output("new-file-list", "children")],
    [Input('new-submit-button', 'n_clicks')],
    [State("adjusted-upload-data", "filename"),
     State("adjusted-upload-data", "contents")]
)
def update_adjust(clicks, filenames, file_contents):
    """Save uploaded files and regenerate the file list."""

    if None in(filenames, file_contents):
        raise PreventUpdate

    else:

        def save_file(name, content):
            """Decode and store a file uploaded with Plotly Dash."""
            if 'cut' in name:

                UPLOAD_DIRECTORY = f"./monthly_data/20/"
            elif '115' in name:
                UPLOAD_DIRECTORY = f"./monthly_data/115/"
                print(UPLOAD_DIRECTORY)

            data = content.encode("utf8").split(b";base64,")[1]
            with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
                fp.write(base64.decodebytes(data))
            print('Saved')

        for name, data in zip(filenames, file_contents):
            save_file(name, data)

        return (None, None, 'Uploaded')


@app.callback(
    Output('month_selection_adjust', 'options'),
    [Input('update_dropdown_adjust', 'n_clicks')])
def update_dropdown_adjust(x):
    return directories_adjust()
