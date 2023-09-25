import os
from calendar import monthrange

# import datetime
from urllib.parse import quote as urlquote
import os
from calendar import monthrange
from urllib.parse import quote as urlquote

import dash.dash_table as dash_table
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output

from app import app, navbar


def get_directories_results(path):
    return [
        {"label": file[:-4], "value": file[:-4]}
        for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file))
        and file.endswith(".csv")
        and file != ".gitkeep"
    ]


def generate_card(content_list):
    return dbc.Card(dbc.CardBody(content_list), className="mt-3")


tab1_content = generate_card(
    [
        html.P(id=f"{id}", className="card-text")
        for id in [
            "MAA_brut",
            "wtc_kWG1TotE_accum",
            "EL115",
            "ELX",
            "ELNX",
            "EL_indefini_left",
            "Epot_eq",
            "EL_Misassigned",
            "EL_PowerRed",
            "ELX%",
            "ELNX%",
        ]
    ]
)

tab2_content = generate_card(
    [
        html.P(id=f"{id}", className="card-text")
        for id in ["Total_duration", "Siemens_duration", "Tarec_duration", "duration_115"]
    ]
)

tabs = dbc.Tabs(
    [
        dbc.Tab(tab1_content, label="Yield Based Availability"),
        dbc.Tab(tab2_content, label="Time Based Availability"),
    ]
)

layout = html.Div(
    [
        navbar,
        dbc.Row(
            [
                dbc.Alert("Please Select a Period", color="warning"),
                dbc.Button("update", id="update_dropdown", style={"display": "none"}),
                dcc.Dropdown(
                    id="month_selection_dropdown",
                    style={
                        "width": "30vw",
                        "textAlign": "center",
                        "margin": "0 auto",
                        "marginBottom": 10,
                    },
                    options=get_directories_results("./monthly_data/results/"),
                ),
                html.A(
                    "Download Detailed results",
                    id="download_button",
                    style={
                        "backgroundColor": "white",
                        "color": "black",
                        "padding": "5px",
                        "textDecoration": "none",
                        "border": "1px solid black",
                    },
                ),
                tabs,
            ],
            className="g-0",
            style={"flexDirection": "column", "alignItems": "center", "justifyContent": "center"},
        ),
        dbc.Row(
            html.A(
                "Download Grouped results",
                id="download_grouped",
                style={
                    "backgroundColor": "white",
                    "color": "black",
                    "padding": "5px",
                    "textDecoration": "none",
                    "border": "1px solid black",
                },
            ),
            className="g-0",
            style={"justifyContent": "center"},
        ),
        dbc.Row(
            id="table", className="g-0", style={"justifyContent": "center", "margin": "0 auto"}
        ),
    ]
)


@app.callback(
    Output("month_selection_dropdown", "options"), [Input("update_dropdown", "n_clicks")]
)
def update_dropdown(n_clicks):
    return get_directories_results("./monthly_data/results/")


@app.callback(
    [
        Output("Total_duration", "children"),
        Output("Siemens_duration", "children"),
        Output("Tarec_duration", "children"),
        Output("MAA_brut", "children"),
        Output("wtc_kWG1TotE_accum", "children"),
        Output("ELX", "children"),
        Output("ELNX", "children"),
        Output("EL_indefini_left", "children"),
        Output("Epot_eq", "children"),
        Output("EL_Misassigned", "children"),
        Output("EL_PowerRed", "children"),
        Output("ELX%", "children"),
        Output("ELNX%", "children"),
        Output("download_button", "href"),
        Output("download_grouped", "href"),
        Output("table", "children"),
    ],
    [Input("month_selection_dropdown", "value")],
)
# @cache.memoize(timeout=60)
def callback_a(x):
    # Danger !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if x is None:
        return tuple(None for i in range(16))

    results = round(
        pd.read_csv(
            f"./monthly_data/results/Grouped_Results/grouped_{x}.csv", decimal=".", sep=","
        ),
        2,
    )

    EL = results["EL"].sum()
    ELX = results["ELX"].sum()
    ELNX = results["ELNX"].sum()
    EL_2006 = results["EL_2006"].sum()
    Ep = results["wtc_kWG1TotE_accum"].sum()
    EL_PowerRed = results["EL_PowerRed"].sum()
    EL_Misassigned = results["EL_Misassigned"].sum()
    EL_indefini_left = results["EL_indefini_left"].sum()

    Epot = results["Epot"].sum()

    ELX_eq = ELX - EL_Misassigned
    ELNX_eq = ELNX + EL_2006 + EL_PowerRed + EL_Misassigned
    Epot_eq = Ep + ELX_eq + ELNX_eq

    EL_wind = results["EL_wind"].sum()
    EL_wind_start = results["EL_wind_start"].sum()
    EL_alarm_start = results["EL_alarm_start"].sum()

    MAA_brut = round(
        100 * (Ep + ELX) / (Ep + ELX + ELNX + EL_2006 + EL_PowerRed),
        2,
    )

    MAA_brut_mis = round(
        100 * (Ep + ELX_eq) / (Epot_eq),
        2,
    )

    Tarec_duration = round(results["Period 1(s)"].sum() / 3600, 2)
    Siemens_duration = round(results["Period 0(s)"].sum() / 3600, 2)
    Total_duration = Tarec_duration + Siemens_duration

    year = int(x[:4])
    month = int(x[5:7])

    days = monthrange(year, month)[1]

    # location = f"/download/{urlquote('results')}/anaconda.exe"
    location = f"/download/results/{urlquote(x)}.csv"

    location_grouped = "/download/results/Grouped_Results/" f"grouped_{urlquote(x)}.csv"

    results = results[
        [
            "Unnamed: 0",
            "StationId",
            "Period 1(s)",
            "Period 0(s)",
            "Duration 115(s)",
            "MAA_brut",
            "MAA_brut_mis",
            "MAA_indefni_adjusted",
        ]
    ]

    table = dash_table.DataTable(
        columns=[{"id": i, "name": i} for i in results.columns],
        data=results.to_dict("records"),
        fixed_rows={"headers": True},
        style_cell={"width": 100},
    )

    # table = html.Iframe(srcDoc=results.to_html())

    return (
        f"""Total duration: {Total_duration} Hours""",
        f"""Siemens duration: {Siemens_duration} Hours""",
        f"""Tarec duration: {Tarec_duration} Hours""",
        f"MAA = {MAA_brut}% | MAA_brut_mis = {MAA_brut_mis}% |",
        f"Energy produced: {Ep:,.2f} kWh",
        f"Energy Lost excusable: {ELX_eq:,.2f} kWh",
        f"Energy Lost non-excusable (+ 2006): {ELNX_eq:,.2f} kWh",
        f"Energy Lost unassigned: {EL_indefini_left:,.2f} kWh",
        f"Potential Energy: {Epot_eq:,.2f} kWh",
        f"EL_Misassigned: {EL_Misassigned:,.2f} kWh",
        f"EL_PowerRed: {EL_PowerRed:,.2f} kWh | EL_2006: {EL_2006:,.2f} kWh",
        f"Energy Lost excusable: {100 * (ELX_eq / (Epot_eq)):,.2f} %",
        f"Energy Lost non-excusable (+ 2006): {100  * ((ELNX_eq) / (Epot_eq)):,.2f} %",
        location,
        location_grouped,
        table,
    )
