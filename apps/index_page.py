import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import os

from app import navbar as navbar
from app import arrow as arrow

directories = [a for a in os.listdir('./monthly_data/')]


column_style = {'height': '100%', 'flexDirection': 'column',
                'display': 'flex', 'alignItems': 'center',
                'justifyContent': 'center',
                'boxShadow': '2px 2px 50px -3px grey',
                'borderRadius': '10px',
                'marginLeft': '5%', 'marginRight': '5%',
                'marginBottom': '1%', 'marginTop': '15%'}

arrow_style = {'height': '100%', 'flexDirection': 'column',
               'display': 'flex', 'alignItems': 'center',
               'justifyContent': 'center',
               'marginLeft': '-10%', 'marginRight': '-10%',
               'marginBottom': '1%', 'marginTop': '15%'}

layout = html.Div([
    navbar,
    dbc.Row([
        dbc.Col(dbc.Button('New Calculation', href='/apps/app1'),
                style=column_style),

        dbc.Col(html.Img(src=arrow,
                         style={'height': '6vh'}),
                style=arrow_style),

        dbc.Col(dbc.Button('Adjust', href='/apps/adjust115'),
                style=column_style),

        dbc.Col(html.Img(src=arrow,
                         style={'height': '6vh'}),
                style=arrow_style),

        dbc.Col(dbc.Button('Results', href='/apps/results'),
                style=column_style)
    ],
        style={'height': '30%', 'display': 'flex', 'alignItems': 'center',
               'justifyContent': 'center'},
        no_gutters=True)
], style={'height': '100vh'})
