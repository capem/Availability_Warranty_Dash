import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash.dash import no_update
# from flask_caching import Cache

import os
from urllib.parse import quote as urlquote
import base64
from datetime import datetime as dt


from app import navbar as navbar
from app import app
import calculation as calculation


# cache = Cache(app.server, config={
#    # try 'filesystem' if you don't want to setup redis
#    'CACHE_TYPE': 'redis',
#    'CACHE_REDIS_URL': os.environ.get('REDIS_URL', '')
# })

directories = [a for a in os.listdir(
    './monthly_data/uploads/') if a != '.gitkeep']


column_style = {'height': '100%', 'flex-direction': 'column',
                'display': 'flex', 'align-items': 'center',
                'justify-content': 'center',
                'boxShadow': '3px 3px 20px -10px #eabf1a',
                'marginLeft': '10%', 'marginRight': '10%',
                'marginBottom': '1%', 'marginTop': '8%'}


layout = html.Div([
    dcc.Store(id='memory'),
    navbar,
    # html.Iframe(src="https://www.youtube.com/embed/1CSeY10zbqo",
    #             style={'width':560 , 'height': 315, 'frameborder': 0}),
    dbc.Row([
        dbc.Col([dbc.Button('Submit/Upload', id='submit-button', n_clicks=0),
                 dbc.Tooltip(
                     "Files To Upload,"
                     "y-m-cnt.zip, y-m-grd.zip,"
                     "y-m-sum.zip, y-m-met.zip,"
                     "y-m-tur.zip",
                     target="submit-button"),

                 dcc.DatePickerSingle(
                     id='date_picker',
                     clearable=True,
                     display_format='YYYY-MM',
                     style={'marginTop': 15},
                     persistence=True),
                 dcc.Loading(dcc.Upload(id="upload-data",
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
                                               "margin": "10px",
                                               },
                                        multiple=False,
                                        max_size=-1),
                             color='#eabf1a'),
                 html.H2("File List"),
                 dcc.Loading(html.Ul(id="file-list"), color='#eabf1a')],
                style=column_style),

        dbc.Col([html.P(children='After Uploading Files'),
                 html.P(children='Enter a value to Calculate'),
                 dcc.Dropdown(id='calculation_selection_dropdown',
                              options=[{'label': i,
                                        'value': i} for i in directories],
                              style={'width': '60%', 'textAlign': 'center',
                                     'margin': '0 auto', 'marginBottom': 10}),
                 dcc.Loading(id='MAA', color='#eabf1a')
                 ], style=column_style)

    ],
        style={'height': '60%', 'display': 'flex', 'align-items': 'center',
               'justify-content': 'center'},
        no_gutters=True)
], style={'height': '100vh'})


def file_download(path, filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = f"/download/uploads/{urlquote(path)}/{urlquote(filename)}"
    return html.A(filename, href=location)


@app.callback(
    [Output("file-list", "children"),
     Output("calculation_selection_dropdown", "options"),
     Output("upload-data", "filename"),
     Output("upload-data", "contents")],
    [Input('submit-button', 'n_clicks')],
    [State('date_picker', 'date'),
     State("upload-data", "filename"),
     State("upload-data", "contents")]
)
def update_output(clicks, date, filenames, file_contents):
    """Save uploaded files and regenerate the file list."""

    if (None in (filenames, file_contents)) and (date is not None):
        print(date)

        year = str(dt.strptime(date, '%Y-%m-%d').year)
        month = str(dt.strptime(date, '%Y-%m-%d').month)

        if len(month) == 1:
            month = '0' + month

        period = year + '-' + month

        UPLOAD_DIRECTORY = f"./monthly_data/uploads/{period}/"
        if not os.path.exists(UPLOAD_DIRECTORY):
            os.makedirs(UPLOAD_DIRECTORY)

        def uploaded_files():
            """List the files in the upload directory."""
            files = []
            for filename in os.listdir(UPLOAD_DIRECTORY):
                path = os.path.join(UPLOAD_DIRECTORY, filename)
                if os.path.isfile(path):
                    files.append(filename)
            return files

        files = uploaded_files()

        return ([html.Li(file_download(period,
                                       filename)) for filename in files],
                no_update,
                no_update,
                no_update,)

    if None in(date, filenames, file_contents):
        raise PreventUpdate

    else:
        year = str(dt.strptime(date, '%Y-%m-%d').year)
        month = str(dt.strptime(date, '%Y-%m-%d').month)

        if len(month) == 1:
            month = '0' + month

        period = year + '-' + month

        UPLOAD_DIRECTORY = f"./monthly_data/uploads/{period}/"

        def save_file(name, content):
            """Decode and store a file uploaded with Plotly Dash."""
            data = content.encode("utf8").split(b";base64,")[1]
            with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
                fp.write(base64.decodebytes(data))

        def uploaded_files():
            """List the files in the upload directory."""
            files = []
            for filename in os.listdir(UPLOAD_DIRECTORY):
                path = os.path.join(UPLOAD_DIRECTORY, filename)
                if os.path.isfile(path):
                    files.append(filename)
            return files

        if not os.path.exists(UPLOAD_DIRECTORY):
            os.makedirs(UPLOAD_DIRECTORY)

        if filenames is not None and file_contents is not None:
            # for name, data in zip(filenames, file_contents):
            save_file(filenames, file_contents)

        files = uploaded_files()

        directories = [a for a in os.listdir(
            './monthly_data/uploads/') if a != '.gitkeep']

        if len(files) == 0:
            return ([html.Li("No files yet!")],
                    [{'label': i, 'value': i} for i in directories])
        else:
            return ([html.Li(file_download(period,
                                           filename)) for filename in files],
                    [{'label': i, 'value': i} for i in directories],
                    None, None)


@app.callback(
    Output('MAA', 'children'),
    [Input('calculation_selection_dropdown', 'value')])
# @cache.memoize(timeout=60)
def callback_calcul(x):
    print(x)
    if x is None:
        raise PreventUpdate

    Results = calculation.full_calculation(period=x)
    Results.to_csv(
        f'./monthly_data/results/{x}-Availability.csv',
        decimal=',', sep=';')

    Results = Results[['StationId', 'wtc_kWG1TotE_accum', 'Epot',
                       'EL', 'EL 115', 'ELX', 'ELNX', 'EL_115_left',
                       'EL_indefini', 'EL_wind', 'EL_wind_start',
                       'EL_alarm_start', 'EL_indefini_left',
                       'Period 1(s)', 'Period 0(s)', 'Duration 115(s)',
                       'Duration 20-25(s)', 'Duration lowind(s)',
                       'Duration lowind_start(s)', 'Duration alarm_start(s)']]

    Results_grouped = round(
        Results.groupby('StationId').sum().reset_index(), 2)

    Ep = Results_grouped['wtc_kWG1TotE_accum']
    EL = Results_grouped['EL']
    ELX = Results_grouped['ELX']
    ELNX = Results_grouped['ELNX']

    EL_wind = Results_grouped['EL_wind']
    EL_wind_start = Results_grouped['EL_wind_start']
    EL_alarm_start = Results_grouped['EL_alarm_start']

    MAA_result = 100 * (Ep + ELX) / (Ep + ELX + ELNX)

    MAA_indefini = 100 * (Ep + ELX) / (Ep + EL)

    MAA_indefni_adjusted = 100 * (
        Ep + ELX + EL_wind + EL_wind_start + EL_alarm_start) / (
            Ep + EL)

    Results_grouped['MAA'] = MAA_result

    Results_grouped['MAA_indefini'] = MAA_indefini

    Results_grouped['MAA_indefni_adjusted'] = MAA_indefni_adjusted

    Results_grouped.index = Results_grouped.index + 1

    Results_grouped.to_csv(
        f"./monthly_data/results/Grouped_Results/grouped_{x}-Availability.csv",
        decimal=',', sep=';')
    print('Done')

    return [dbc.NavLink("Go To Results",
                        href='/apps/results',)]
