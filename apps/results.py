import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_table

import pandas as pd
import os
# import datetime
from urllib.parse import quote as urlquote
# from dash.exceptions import PreventUpdate

from app import navbar
from app import app

from calendar import monthrange

# from flask_caching import Cache
# from dash.dash import no_update

# cache = Cache(app.server, config={
#    # try 'filesystem' if you don't want to setup redis
#    'CACHE_TYPE': 'redis',
#    'CACHE_REDIS_URL': os.environ.get('REDIS_URL', '')
# })


def directories_results():

    directories_results = [a for a in os.listdir(
        './monthly_data/results/') if a != '.gitkeep']
    return([{'label': i, 'value': i} for i in directories_results])


# tab_style = {'width': '45vw', 'height': '60%'}

tab1_content = dbc.Card(
    dbc.CardBody(
        [
            html.P(id='MAA_result', className="card-text"),
            html.P(id='wtc_kWG1TotE_accum', className="card-text"),
            html.P(id='EL115', className="card-text"),
            html.P(id='ELX', className="card-text"),
            html.P(id='ELNX', className="card-text"),
            html.P(id='EL_indefini_left', className="card-text"),
            html.P(id='Epot', className="card-text"),
            html.P(id='EL_Misassigned', className='card-text'),
            html.P(id='EL_PowerRed', className='card-text'),
            html.P(id='ELX%', className="card-text"),
            html.P(id='ELNX%', className="card-text"),
            # dbc.Button("Don't click here", color="danger"),

        ]
    ),
    className="mt-3",
    # style=tab_style
)

tab2_content = dbc.Card(
    dbc.CardBody(
        [html.P(id='Total_duration', className="card-text"),
         html.P(id='Siemens_duration', className="card-text"),
         html.P(id='Tarec_duration', className="card-text"),
         html.P(id='duration_115', className="card-text"),
            # dbc.Button("Click here", color="success"),
         ]
    ),
    className="mt-3",
    # style=tab_style
)

tabs = dbc.Tabs(
    [
        dbc.Tab(tab1_content, label="Yield Based Availability"),
        dbc.Tab(tab2_content, label="Time Based Availability"),
    ]
)


layout = html.Div([
    navbar,

    dbc.Row([dbc.Alert("Please Select a Period", color="warning"),
             dbc.Button('update', id='update_dropdown',
                        style={'display': 'none'}),
             dcc.Dropdown(id='month_selection_dropdown',
                          style={'width': '30vw', 'textAlign': 'center',
                                 'margin': '0 auto', 'marginBottom': 10},
                          options=directories_results()),
             html.A('Download Detailed Results', id="download_button",
                    style={'backgroundColor': 'white',
                           'color': 'black',
                           'padding': '5px',
                           'textDecoration': 'none',
                           'border': '1px solid black',
                           }),
             tabs],
            no_gutters=True,
            style={'flexDirection': 'column', 'alignItems': 'center',
                   'justifyContent': 'center'}),

    dbc.Row(html.A('Download Grouped Results', id="download_grouped",
                   style={'backgroundColor': 'white',
                          'color': 'black',
                          'padding': '5px',
                          'textDecoration': 'none',
                          'border': '1px solid black',
                          }),
            no_gutters=True, style={'justifyContent': 'center'}),
    dbc.Row(id='table', no_gutters=True,
            style={
                'justifyContent': 'center',
                'margin': '0 auto'
            }
            )
])


@app.callback(
    Output('month_selection_dropdown', 'options'),
    [Input('update_dropdown', 'n_clicks')])
def update_dropdown(x):
    path = './monthly_data/results/'
    directories_results = [a for a in os.listdir(
        path) if (os.path.isfile(os.path.join(path, a)) and a != '.gitkeep' and a.endswith('.csv'))]
    return([{'label': i[:-4], 'value': i[:-4]} for i in directories_results])


@app.callback(
    [Output('Total_duration', 'children'),
     Output('Siemens_duration', 'children'),
     Output('Tarec_duration', 'children'),
     Output('duration_115', 'children'),
     Output('MAA_result', 'children'),
     Output('wtc_kWG1TotE_accum', 'children'),
     Output('EL115', 'children'),
     Output('ELX', 'children'),
     Output('ELNX', 'children'),
     Output('EL_indefini_left', 'children'),
     Output('Epot', 'children'),
     Output('EL_Misassigned', 'children'),
     Output('EL_PowerRed', 'children'),
     Output('ELX%', 'children'),
     Output('ELNX%', 'children'),

     Output('download_button', 'href'),
     Output('download_grouped', 'href'),
     Output('table', 'children')
     ],
    [Input('month_selection_dropdown', 'value')])
# @cache.memoize(timeout=60)
def callback_a(x):

    # Danger !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if x is None:
        return tuple(None for i in range(18))

    Results = round(pd.read_csv(
        f"./monthly_data/results/Grouped_Results/grouped_{x}.csv",
        decimal=',', sep=';'), 2)

    Ep = Results['wtc_kWG1TotE_accum'].sum()
    ELX = Results['ELX'].sum()
    ELNX = Results['ELNX'].sum()
    Epot = Results['Epot'].sum()
    EL = Results['EL'].sum()
    EL_2006 = Results['EL_2006'].sum()
    EL_indefini_left = Results['EL_indefini_left'].sum()

    EL_Misassigned = Results['EL_Misassigned'].sum()
    EL_PowerRed = Results['EL_PowerRed'].sum()

    EL_wind = Results['EL_wind'].sum()
    EL_wind_start = Results['EL_wind_start'].sum()
    EL_alarm_start = Results['EL_alarm_start'].sum()

    MAA_result = round(100 * (Ep + ELX) / (Ep + ELX + ELNX + EL_2006), 2)

    MAA_indefini = round(100 * (Ep + ELX) / (Ep + EL), 2)

    MAA_indefini_adjusted = round(100 * (
        Ep + ELX) / (
            Ep + EL - (EL_wind + EL_wind_start + EL_alarm_start)), 2)

    Siemens_duration = Results['Period 1(s)'].sum() / 3600
    Tarec_duration = Results['Period 0(s)'].sum() / 3600
    Total_duration = Siemens_duration + Tarec_duration
    duration_115 = Results['Duration 115(s)'].sum() / 3600

    year = int(x[:4])
    month = int(x[5:7])

    days = monthrange(year, month)[1]

    # location = f"/download/{urlquote('results')}/anaconda.exe"
    location = f"/download/results/{urlquote(x)}.csv"

    location_grouped = ('/download/results/Grouped_Results/'
                        f'grouped_{urlquote(x)}.csv')

    Results = Results[['Unnamed: 0', 'StationId', 'Period 1(s)',
                       'Period 0(s)', 'Duration 115(s)', 'MAA',
                       'MAA_indefini', 'MAA_indefni_adjusted']]

    table = dash_table.DataTable(
        columns=[
            {"id": i, "name": i} for i in Results.columns],
        data=Results.to_dict('records'),
        fixed_rows={'headers': True},
        style_cell={
            'width': 100
        }
    )

    # table = html.Iframe(srcDoc=Results.to_html())

    return (f'''Total duration: {round(Total_duration, 2)} Hour |
                {round(100*Total_duration/(24*days*131),2)}%''',

            f'''Siemens duration: {round(Siemens_duration, 2)} Hour |
                {round(100*Siemens_duration/(24*days*131),2)}%''',

            f'''Tarec duration: {round(Tarec_duration, 2)} Hour |
                {round(100*Tarec_duration/(24*days*131), 2)}%''',

            f'''115 duration: {round(duration_115, 2)} Hour |
                {round(100*duration_115/(24*days*131), 2)}%''',

            f'MAA = {MAA_result}% | MAA_indefini = {MAA_indefini}% |',
            f'MAA_indefini_adjusted = {MAA_indefini_adjusted}%',
            f'Energy produced: {Ep:,.2f} kWh',
            f'Energy Lost excusable: {ELX:,.2f} kWh',
            f'Energy Lost non-excusable: {ELNX + EL_2006:,.2f} kWh',
            f'Energy Lost unassigned: {EL_indefini_left:,.2f} kWh',
            f'Potential Energy: {Epot:,.2f} kWh',

            f'EL_Misassigned: {EL_Misassigned:,.2f} kWh',
            f'EL_PowerRed: {EL_PowerRed:,.2f} kWh',

            f'Energy Lost excusable: {100 * (ELX / (ELX + ELNX + EL_2006 + Ep)):,.2f} %',
            f'Energy Lost non-excusable: {100  * (ELNX / (ELX + ELNX + EL_2006 + Ep)):,.2f} %',

            location,
            location_grouped,
            table
            )
