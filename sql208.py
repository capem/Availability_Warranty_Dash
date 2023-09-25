import pyodbc
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime as dt


def construct_query(period_start, period_end, alarms_0_1):
    # Constructing the IN clause for the SQL query
    in_clause = ", ".join(["?"] * len(alarms_0_1))

    query = f"""
    SELECT [TimeOn], [TimeOff], [StationNr], [Alarmcode], [Parameter], [ID]
    FROM [TARECREPORTING].[dbo].[tblAlarmLog]
    WHERE (
      ([TimeOff] BETWEEN ? AND ?)
      OR ([TimeOn] BETWEEN ? AND ?)
      OR ([TimeOn] <= ? AND [TimeOff] >= ?)
      OR ([TimeOn] <= ? AND [TimeOff] >= ?)
      OR ([TimeOn] <= ? AND [TimeOff] IS NULL AND [Alarmcode] IN ({in_clause})))
    """

    # Flatten the params list and add alarms_0_1 values at the end
    params = [
        period_start,
        period_end,
        period_start,
        period_end,
        period_start,
        period_end,
        period_start,
        period_start,
        period_end,
    ] + alarms_0_1.tolist()

    return query, params


def export_to_rpt(df):
    df.to_csv("output.rpt", sep="|", index=False)


def main():
    # Define connection parameters
    driver = "{ODBC Driver 11 for SQL Server}"
    server = "192.168.0.208"
    database = "master"
    username = "ReadOnlySA"
    password = "T@rec2016"
    connection_string = (
        f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}"
    )

    # Define the period and derive necessary date/time variables
    period = "2023-09"

    period_dt = dt.strptime(period, "%Y-%m")
    period_year = period_dt.year

    next_period_dt = period_dt + relativedelta(months=+1)
    next_period = next_period_dt.strftime("%Y-%m")

    period_start = pd.Timestamp(f"{period}-01 00:00:00.000")
    period_end = pd.Timestamp(f"{next_period}-01 00:00:00.000")

    error_list = pd.read_excel(r"Alarmes List Norme RDS-PP_Tarec.xlsx")
    error_list.Number = error_list.Number.astype(int)  # ,errors='ignore'
    error_list.drop_duplicates(subset=["Number"], inplace=True)
    error_list.rename(columns={"Number": "Alarmcode"}, inplace=True)
    alarms_0_1 = error_list.loc[error_list["Error Type"].isin([1, 0])].Alarmcode

    try:
        with pyodbc.connect(connection_string) as connection:
            query, params = construct_query(period_start, period_end, alarms_0_1)
            df = pd.read_sql(query, connection, params=params)
        export_to_rpt(df)
        print("Data exported successfully to output.rpt!")
    except pyodbc.Error as e:
        print(f"Failed to fetch and export data. Reason: {str(e)}")


if __name__ == "__main__":
    main()
