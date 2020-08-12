import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash.dash import no_update
import re
# from flask_caching import Cache

import os
from urllib.parse import quote as urlquote
import base64
from datetime import datetime as dt


from app import navbar as navbar
from app import app
import calculation as calculation

import dash_uploader as du

temp_upload_directory = r"./monthly_data/uploads/temp/"

temp_upload_directory_abs = os.path.abspath(
    temp_upload_directory).replace('\\', '/') + '/'

du.configure_upload(app, temp_upload_directory, use_upload_id=False)
# cache = Cache(app.server, config={
#    # try 'filesystem' if you don't want to setup redis
#    'CACHE_TYPE': 'redis',
#    'CACHE_REDIS_URL': os.environ.get('REDIS_URL', '')
# })

directories = [a for a in os.listdir(
    './monthly_data/uploads/') if a != '.gitkeep']


column_style = {'height': '60vh',
                'gridTemplateColumns': '100%',
                'display': 'grid',
                'alignItems': 'center',
                'justifyContent': 'center',
                'boxShadow': '3px 3px 20px -10px #eabf1a',
                'overflow': 'visible',
                'margin': '5%',
                'position': 'unset'}

layout = html.Div([
    dcc.Store(id='memory'),
    navbar,
    # html.Iframe(src="https://www.youtube.com/embed/1CSeY10zbqo",
    #             style={'width':560 , 'height': 315, 'frameborder': 0}),
    dbc.Row([

        dbc.Col([

                dbc.Tooltip('''Files To Upload,
                             y-m-cnt.zip, y-m-grd.zip,
                             y-m-sum.zip, y-m-met.zip,
                             y-m-tur.zip''',
                            target="dash_uploader"
                            ),

                dcc.DatePickerSingle(id='date_picker',
                                     clearable=False,
                                     display_format='YYYY-MM',
                                     style={'margin': 15,
                                            'position': 'static',
                                            'placeSelf': 'center'},
                                     persistence=True,
                                     number_of_months_shown=1,
                                     show_outside_days=False,
                                     with_portal=True
                                     ),

                html.H3("Uploaded Files List",
                        style={'placeSelf': 'center'}),
                dcc.Loading(html.Ul(id="file-list"),
                            color='#eabf1a'),

                html.Div(
                    du.Upload(id='dash_uploader',
                              text='Drag and Drop Here to upload!',
                              max_files=1,
                              filetypes=['zip'],
                              default_style={'overflow': 'hide'}

                              ),
                    style={}
                )],
                style=dict(column_style, overflow='auto')
                ),

        dbc.Col([html.P(['After Uploading Files', html.Br(),
                         'Select a month to Calculate'],
                        style={'placeSelf': 'end center'}),
                 html.Div(  # wrapped because of styling problmes in dropdown
                     dcc.Dropdown(id='calculation_selection_dropdown',
                                  options=[{'label': i,
                                            'value': i} for i in directories],
                                  style={}),
                     style={'width': '60%', 'placeSelf': 'center'},
        ),
            dcc.Loading(id='MAA', color='#eabf1a')
        ], style=column_style
        )

    ],
        align='center',
        justify='center',
        no_gutters=True
    )],
    style={'height': '100vh',
           'display': 'flex',
           'flexDirection': 'column',
           'alignContent': 'stretch',
           'justifyItems': 'stretch'}
)


def file_download(path, filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = f"/download/uploads/{urlquote(path)}/{urlquote(filename)}"
    return html.A(filename, href=location)


@app.callback(
    [Output("file-list", "children"),
     Output("calculation_selection_dropdown", "options")],
    [Input('date_picker', 'date'),
     Input('dash_uploader', 'isCompleted')],
    [State('dash_uploader', 'fileNames')]
)
def update_output(date, isCompleted, fileNames):
    """regenerate the file list."""
    if date is None:
        raise PreventUpdate

    year = str(dt.strptime(date, '%Y-%m-%d').year)
    month = str(dt.strptime(date, '%Y-%m-%d').month)

    period = year + '-' + month.zfill(2)

    upload_directory = f"./monthly_data/uploads/{period}/"

    def uploaded_files():
        """List the files in the upload directory."""
        files = []
        try:
            for filename in os.listdir(upload_directory):
                path = os.path.join(upload_directory, filename)
                if os.path.isfile(path) & filename.lower().endswith('.zip'):
                    files.append(filename)
        except FileNotFoundError:
            pass

        return files

    def move_uploaded_file(fileNames):

        file_regex = r"^20[1-9][0-9]-(0[1-9]|1[0-2])-(cnt|sum|tur|grd|met).zip"

        if not re.match(file_regex, fileNames[0]):
            print('Err: File name not matching regex ,file will be deleted.')
            os.remove(temp_upload_directory + fileNames[0])
        else:
            move_to_directory = f"./monthly_data/uploads/{period}/"
            # if not os.path.exists(move_to_directory):
            #     os.makedirs(move_to_directory)

            os.renames(temp_upload_directory + fileNames[0],
                       move_to_directory + fileNames[0])

            print(fileNames)
            print(fileNames[0][:7])

    if isCompleted:
        move_uploaded_file(fileNames)

    directories = [a for a in os.listdir(
        './monthly_data/uploads/') if (a not in ['temp', '.gitkeep'])]

    files = uploaded_files()

    if len(files) == 0:
        return ([html.Li("No files yet!")],
                [{'label': i, 'value': i} for i in directories])
    else:
        return ([html.Li(file_download(period,
                                       filename)) for filename in files],
                [{'label': i, 'value': i} for i in directories])


@app.callback(
    Output('MAA', 'children'),
    [Input('calculation_selection_dropdown', 'value')])
# @cache.memoize(timeout=60)
def callback_calcul(value):

    if value is None:
        raise PreventUpdate

    try:
        Results = calculation.full_calculation(period=value)
    except FileNotFoundError:
        return html.P('Please upload rest of files.')

    Results.to_csv(
        f'./monthly_data/results/{value}-Availability.csv',
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
        Ep + ELX) / (
            Ep + EL - (EL_wind + EL_wind_start + EL_alarm_start))

    Results_grouped['MAA'] = MAA_result

    Results_grouped['MAA_indefini'] = MAA_indefini

    Results_grouped['MAA_indefni_adjusted'] = MAA_indefni_adjusted

    Results_grouped.index = Results_grouped.index + 1

    Results_grouped.to_csv(
        f"./monthly_data/results/Grouped_Results/grouped_{value}-Availability.csv",
        decimal=',', sep=';')
    print('Done')

    return [dbc.NavLink("Go To Results",
                        href='/apps/results',)]
