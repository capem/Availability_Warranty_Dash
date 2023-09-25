import calculation

from datetime import datetime, timedelta

# Define the start and end dates
start_date = datetime(2019, 1, 1)
end_date = datetime(2019, 12, 1)

# Define the increment for each iteration (1 month)
month_increment = timedelta(days=30)

# Initialize a list to store the date strings
date_strings = []

# Start the loop
current_date = start_date
while current_date <= end_date:
    date_string = current_date.strftime("%Y-%m")
    date_strings.append(date_string)
    current_date += month_increment

for date_string in date_strings:
    Results = calculation.full_calculation(date_string)
    Results = Results[
        [
            "StationId",
            "wtc_kWG1TotE_accum",
            "Epot",
            "EL",
            "EL 115",
            "ELX",
            "ELNX",
            "EL_115_left",
            "EL_indefini",
            "EL_wind",
            "EL_wind_start",
            "EL_alarm_start",
            "EL_indefini_left",
            "Period 1(s)",
            "Period 0(s)",
            "Duration 115(s)",
            "Duration 20-25(s)",
            "Duration lowind(s)",
            "EL_2006",
            "Duration lowind_start(s)",
            "Duration alarm_start(s)",
            "EL_Misassigned",
            "EL_PowerRed",
        ]
    ]

    Results_grouped = round(Results.groupby("StationId").sum().reset_index(), 2)

    Ep = Results_grouped["wtc_kWG1TotE_accum"]
    EL = Results_grouped["EL"]
    ELX = Results_grouped["ELX"]
    ELNX = Results_grouped["ELNX"]
    EL_2006 = Results_grouped["EL_2006"]
    EL_PowerRed = Results_grouped["EL_PowerRed"]
    EL_Misassigned = Results_grouped["EL_Misassigned"]

    ELX_eq = ELX - EL_Misassigned
    ELNX_eq = ELNX + EL_2006 + EL_PowerRed + EL_Misassigned
    Epot_eq = Ep + ELX_eq + ELNX_eq

    EL_wind = Results_grouped["EL_wind"]
    EL_wind_start = Results_grouped["EL_wind_start"]
    EL_alarm_start = Results_grouped["EL_alarm_start"]

    Results_grouped["MAA_brut"] = 100 * (Ep + ELX) / (Ep + ELX + ELNX + EL_2006 + EL_PowerRed)

    Results_grouped["MAA_brut_mis"] = round(
        100 * (Ep + ELX_eq) / (Epot_eq),
        2,
    )

    Results_grouped["MAA_indefni_adjusted"] = (
        100 * (Ep + ELX) / (Ep + EL - (EL_wind + EL_wind_start + EL_alarm_start))
    )

    Results_grouped.index = Results_grouped.index + 1

    Results_grouped.to_csv(
        f"./monthly_data/results/Grouped_Results/grouped_{date_string}-Availability.csv",
        decimal=".",
        sep=",",
    )
