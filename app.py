#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 16:45:52 2020

@author: sd
"""
import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
from flask import send_from_directory


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

PLOTLY_LOGO = "https://i.ibb.co/3v8qBHx/tarec-warranty.png"

# server = Flask(__name__)
# app = dash.Dash(server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server


UPLOAD_DIRECTORY = "./monthly_data/"


@app.server.route("/download/<path:filepath>")
def download(filepath):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, filepath, as_attachment=True)


app.config.suppress_callback_exceptions = True


PLOTLY_LOGO = app.get_asset_url("tarec_warranty.png")
arrow = app.get_asset_url("forward_arrow.png")

navbar = dbc.Navbar(
    [
        html.A(
            html.Img(
                src=PLOTLY_LOGO,
                style={"height": "45px",
                       "boxShadow": "3px 3px 20px -10px blue"},
            ),
            href="/",
        ),
        dbc.NavLink("Calculate", href="/apps/app1",
                    style={"fontSize": "150%"}),
        dbc.NavLink("Adjust", href="/apps/adjust115",
                    style={"fontSize": "150%"}),
        dbc.NavLink("Results", href="/apps/results",
                    style={"fontSize": "150%"}),
    ],
    color="#eabf1a",
    dark=True,
    style={"justifyContent": "start",
           "flexWrap": "nowrap",
           "width": "100%"},
)

# import index
