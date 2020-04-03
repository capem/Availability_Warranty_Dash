import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_table

import pandas as pd
import os
import datetime
from urllib.parse import quote as urlquote
from dash.exceptions import PreventUpdate

from app import navbar
from app import app

from calendar import monthrange

from flask_caching import Cache
from dash.dash import no_update

# cache = Cache(app.server, config={
#    # try 'filesystem' if you don't want to setup redis
#    'CACHE_TYPE': 'redis',
#    'CACHE_REDIS_URL': os.environ.get('REDIS_URL', '')
# })


def directories_results():

    directories_results = [a for a in os.listdir(
        './monthly_data/results/') if a != '.gitkeep']
    return([{'label': i, 'value': i} for i in directories_results])


tab_style = {'width': '35vw', 'height': '35vh'}

tab1_content = dbc.Card(
    dbc.CardBody(
        [
            html.P(id='MAA_result', className="card-text"),
            html.P(id='wtc_kWG1Tot_accum', className="card-text"),
            html.P(id='EL115', className="card-text"),
            html.P(id='ELX', className="card-text"),
            html.P(id='ELNX', className="card-text"),
            html.P(id='EL_indefni', className="card-text"),
            html.P(id='Epot', className="card-text"),
            # dbc.Button("Don't click here", color="danger"),

        ]
    ),
    className="mt-3", style=tab_style
)

tab2_content = dbc.Card(
    dbc.CardBody(
        [html.P(id='Total_duration', className="card-text"),
            html.P(id='Siemens_duration', className="card-text"),
            html.P(id='Tarec_duration', className="card-text"),
            # dbc.Button("Click here", color="success"),
         ]
    ),
    className="mt-3", style=tab_style
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
                    style={'background-color': 'white',
                           'color': 'black',
                           'padding': '5px',
                           'text-decoration': 'none',
                           'border': '1px solid black',
                           }),
             tabs],
            no_gutters=True,
            style={'flex-direction': 'column', 'align-items': 'center',
                   'justify-content': 'center'}),
    dbc.Row(html.A('Download Grouped Results', id="download_grouped",
                   style={'background-color': 'white',
                          'color': 'black',
                          'padding': '5px',
                          'text-decoration': 'none',
                          'border': '1px solid black',
                          }),
            no_gutters=True, style={'justify-content': 'center'}),
    dbc.Row(id='table', no_gutters=True, style={'justify-content': 'center'})
])


@app.callback(
    Output('month_selection_dropdown', 'options'),
    [Input('update_dropdown', 'n_clicks')])
def update_dropdown(x):
    path = './monthly_data/results/'
    directories_results = [a for a in os.listdir(
        path) if (os.path.isfile(os.path.join(path, a)) and a != '.gitkeep')]
    return([{'label': i[:-4], 'value': i[:-4]} for i in directories_results])


@app.callback(
    [Output('Total_duration', 'children'),
     Output('Siemens_duration', 'children'),
     Output('Tarec_duration', 'children'),
     Output('MAA_result', 'children'),
     Output('wtc_kWG1Tot_accum', 'children'),
     Output('EL115', 'children'),
     Output('ELX', 'children'),
     Output('ELNX', 'children'),
     Output('EL_indefini', 'children'),
     Output('Epot', 'children'),
     Output('download_button', 'href'),
     Output('download_grouped', 'href'),
     Output('table', 'children')],
    [Input('month_selection_dropdown', 'value')])
# @cache.memoize(timeout=60)
def callback_a(x):

    if x is None:
        return tuple(None for i in range(13))

    Results = pd.read_csv(
        f"./monthly_data/results/Grouped_Results/grouped_{x}.csv",
        decimal=',', sep=';')
    
    location_grouped = (f"/download/results/Grouped_Results/"
                        f"grouped_{urlquote(x)}.csv")

    Ep = Results['wtc_kWG1Tot_accum'].sum()
    ELX = Results['ELX'].sum()
    ELNX = Results['ELNX'].sum()
    Epot = Results['Epot'].sum()
    EL115 = Results['EL 115'].sum()
    EL_indefini = Results['EL_indefini'].sum()

    MAA_result = round(100 * (Ep + ELX) / (Ep + ELX + ELNX), 2)

    MAA_115 = round(100 * (Ep + ELX) / (Ep + EL115), 2)

    Siemens_duration = Results['Period 1(s)'].sum() / 3600
    Tarec_duration = Results['Period 0(s)'].sum() / 3600
    Total_duration = Siemens_duration + Tarec_duration

    year = int(x[:4])
    month = int(x[5:7])

    days = monthrange(year, month)[1]

    # location = f"/download/{urlquote('results')}/anaconda.exe"
    location = f"/download/{urlquote('results')}/{urlquote(x)}"

    table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in Results.columns],
        data=Results.to_dict('records'))

    return (f'Total duration: {round(Total_duration, 2)} Hour | \
                {round(100*Total_duration/(24*days*131),2)}%',
            f'Siemens duration: {round(Siemens_duration, 2)} Hour | \
                {round(100*Siemens_duration/(24*days*131),2)}%',
            f'Tarec duration: {round(Tarec_duration, 2)} Hour | \
                {round(100*Tarec_duration/(24*days*131), 2)}%',

            f'MAA = {MAA_result}% | MAA_115 = {MAA_115}% |',
            f'EL115 = {EL115:,.2f} kWh',
            f'Energy produced: {Ep:,.2f} kWh',
            f'Energy Lost excusable: {ELX:,.2f} kWh',
            f'Energy Lost non-excusable: {ELNX:,.2f} kWh',
            f'Energy Lost unassigned: {EL_indefini:,.2f} kWh',
            f'Potential Energy: {Epot:,.2f} kWh',
            location,
            location_grouped,
            table
            )
