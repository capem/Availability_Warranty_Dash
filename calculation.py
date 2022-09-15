from operator import index
from dateutil.relativedelta import relativedelta
from datetime import datetime as dt
from pandas.core.indexing import is_nested_tuple
import pyodbc
from zipfile import ZipFile
import os
import numpy as np
import multiprocessing as mp
import pandas as pd
import urllib
from sqlalchemy import create_engine

from scipy.stats import binned_statistic

# import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import integrate
from scipy.interpolate import interp1d


def zip_to_df(data_type, sql, period):

    file_name = f"{period}-{data_type.lower()}"

    data_type_path = f"./monthly_data/uploads/{data_type.upper()}/"

    ZipFile(f"{data_type_path}{file_name}.zip", "r").extractall(data_type_path)

    conn_str = (
        r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
        rf"DBQ={data_type_path}{file_name}.mdb;"
    )
    conn_str = f"access+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn_str)}"
    cnxn = create_engine(conn_str, echo=True)

    df = pd.read_sql(sql, cnxn)
    cnxn.dispose()
    return df


def sqldate_to_datetime(column):

    try:
        column = column.str.replace(",", ".").astype(float)
    except:
        pass
    day_parts = np.modf(column.loc[~column.isna()])

    column.loc[~column.isna()] = (
        dt(1899, 12, 30)
        + day_parts[1].astype("timedelta64[D]", errors="ignore")
        + (day_parts[0] * 86400000).astype("timedelta64[ms]", errors="ignore")
    )
    column = column.fillna(pd.NaT)
    return column


# Determine alarms real periods
def cascade(df):

    df.reset_index(inplace=True, drop=True)
    df["TimeOffMax"] = df.TimeOff.cummax().shift()

    df.at[0, "TimeOffMax"] = df.at[0, "TimeOn"]

    return df


# looping through turbines and applying cascade method
def apply_cascade(result_sum):

    # Sort by alarm ID
    result_sum.sort_values(["TimeOn", "ID"], inplace=True)
    df = result_sum.groupby("StationNr").apply(cascade)

    mask_root = df.TimeOn.values >= df.TimeOffMax.values
    mask_children = (df.TimeOn.values < df.TimeOffMax.values) & (
        df.TimeOff.values > df.TimeOffMax.values
    )
    mask_embedded = df.TimeOff.values <= df.TimeOffMax.values

    df.loc[mask_root, "NewTimeOn"] = df.loc[mask_root, "TimeOn"]
    df.loc[mask_children, "NewTimeOn"] = df.loc[mask_children, "TimeOffMax"]
    df.loc[mask_embedded, "NewTimeOn"] = df.loc[mask_embedded, "TimeOff"]

    # df.drop(columns=["TimeOffMax"], inplace=True)

    df.reset_index(inplace=True, drop=True)

    TimeOff = df.TimeOff
    NewTimeOn = df.NewTimeOn

    df["RealPeriod"] = abs(TimeOff - NewTimeOn)

    mask_siemens = df["Error Type"] == 1
    mask_tarec = df["Error Type"] == 0

    df["Period Siemens(s)"] = df[mask_siemens].RealPeriod  # .dt.seconds
    df["Period Tarec(s)"] = df[mask_tarec].RealPeriod  # .dt.seconds

    return df


def realperiod_10mins(last_df, type="1-0"):

    last_df["TimeOnRound"] = last_df["NewTimeOn"].dt.ceil("10min")
    last_df["TimeOffRound"] = last_df["TimeOff"].dt.ceil("10min")
    last_df["TimeStamp"] = last_df.apply(
        lambda row: pd.date_range(row["TimeOnRound"], row["TimeOffRound"], freq="10min"), axis=1,
    )
    last_df = last_df.explode("TimeStamp")
    if type != "2006":
        last_df["RealPeriod"] = pd.Timedelta(0)
        last_df["Period Siemens(s)"] = pd.Timedelta(0)
        last_df["Period Tarec(s)"] = pd.Timedelta(0)

    df_TimeOn = last_df[["TimeStamp", "NewTimeOn"]].copy()
    df_TimeOff = last_df[["TimeStamp", "TimeOff"]].copy()

    df_TimeOn.loc[:, "TimeStamp"] = df_TimeOn["TimeStamp"] - pd.Timedelta(minutes=10)

    last_df["10minTimeOn"] = df_TimeOn[["TimeStamp", "NewTimeOn"]].max(1).values

    last_df["10minTimeOff"] = df_TimeOff[["TimeStamp", "TimeOff"]].min(1).values

    last_df["RealPeriod"] = last_df["10minTimeOff"] - last_df["10minTimeOn"]

    if type != "2006":
        mask_siemens = last_df["Error Type"] == 1
        mask_tarec = last_df["Error Type"] == 0
        last_df.loc[mask_siemens, "Period Siemens(s)"] = last_df.loc[mask_siemens, "RealPeriod"]
        last_df.loc[mask_tarec, "Period Tarec(s)"] = last_df.loc[mask_tarec, "RealPeriod"]

    return last_df


def remove_1005_overlap(df):  # input => alarmsresultssum

    df = df[
        [
            "TimeOn",
            "TimeOff",
            "StationNr",
            "Alarmcode",
            "Parameter",
            "ID",
            "NewTimeOn",
            "OldTimeOn",
            "OldTimeOff",
            "UK Text",
            "Type Selected",
            "Error Type",
            "RealPeriod",
        ]
    ].copy()
    idx_to_drop = df.loc[(df.RealPeriod == pd.Timedelta(0)) & (df.Alarmcode != 1005)].index
    df.drop(idx_to_drop, inplace=True)

    df.reset_index(drop=True, inplace=True)
    df_1005 = df.query("Alarmcode == 1005")

    df["TimeOn"] = df["NewTimeOn"]

    for _, j in df_1005.iterrows():

        overlap_end = (
            (df["TimeOn"] <= j["TimeOn"])
            & (df["TimeOn"] <= j["TimeOff"])
            & (df["TimeOff"] > j["TimeOn"])
            & (df["TimeOff"] <= j["TimeOff"])
            & (df["StationNr"] == j["StationNr"])
            & (df["Alarmcode"] != 1005)
        )

        overlap_start = (
            (df["TimeOn"] >= j["TimeOn"])
            & (df["TimeOn"] <= j["TimeOff"])
            & (df["TimeOff"] > j["TimeOn"])
            & (df["TimeOff"] >= j["TimeOff"])
            & (df["StationNr"] == j["StationNr"])
            & (df["Alarmcode"] != 1005)
        )

        embedded = (
            (df["TimeOn"] < j["TimeOn"])
            & (df["TimeOff"] > j["TimeOff"])
            & (df["StationNr"] == j["StationNr"])
            & (df["Alarmcode"] != 1005)
        )
        df_helper = df.loc[embedded].copy()

        df.loc[overlap_start, "TimeOn"] = j["TimeOff"]
        df.loc[overlap_end, "TimeOff"] = j["TimeOn"]

        # ---------------------------------------------------
        if embedded.sum():
            df.loc[embedded, "TimeOff"] = j["TimeOn"]

            df_helper["TimeOn"] = j["TimeOff"]
            df = pd.concat([df, df_helper]).sort_values(["TimeOn", "ID"])

        # ---------------------------------------------------
        # df.reset_index(drop=True, inplace=True)
        reverse_embedded = (
            (df["TimeOn"] >= j["TimeOn"])
            & (df["TimeOff"] <= j["TimeOff"])
            & (df["StationNr"] == j["StationNr"])
            & (df["Alarmcode"] != 1005)
        )
        if reverse_embedded.sum():
            # df.drop(df.loc[reverse_embedded].index, inplace=True)
            # df = df.loc[~df.index.drop(df.loc[reverse_embedded].index)]
            df.loc[reverse_embedded, "TimeOn"] = df.loc[reverse_embedded, "TimeOff"]

    df.loc[df["Alarmcode"] == 1005, "TimeOn"] = df.loc[df["Alarmcode"] == 1005, "OldTimeOn"]

    df = apply_cascade(df)

    return df


def full_range(df, full_range_var):

    new_df = pd.DataFrame(index=full_range_var)

    df = df.set_index("TimeStamp")
    df = df.drop("StationNr", axis=1)

    return new_df.join(df, how="left")


def CF(M, WTN=131, AL_ALL=0.08):
    def AL(M):
        return AL_ALL * (M - 1) / (WTN - 1)

    return (1 - AL_ALL) / (1 - AL(M))


def ep_cf(x):
    M = len(x)
    x = x.mean()
    x = round(x * CF(M), 2)
    return x


def cf_column(x):
    M = len(x)
    return CF(M)


def Epot_case_2(df):
    CB2 = pd.read_excel("CB2.xlsx")
    CB2 = CB2.astype(int).drop_duplicates()
    CB2_interp = interp1d(CB2.Wind, CB2.Power, kind="linear", fill_value="extrapolate")

    Epot = df.apply(lambda x: float(CB2_interp(x.wtc_AcWindSp_mean)), axis=1).values / 6

    return Epot


def Epot_case_3(period):
    NWD = pd.read_excel("NWD.xlsx", index_col=0)
    SWF = pd.read_excel("SWF.xlsx", index_col=0)
    CB2 = pd.read_excel("CB2.xlsx")
    PWE = 0.92
    NAE = 0
    CB2 = CB2.astype(int).drop_duplicates()
    CB2_interp = interp1d(CB2.Wind, CB2.Power, kind="linear", fill_value="extrapolate")

    bins_v = np.arange(1, 26, 1)
    for v in bins_v:
        NAE += CB2_interp(v) * NWD.loc[v].values[0]

    NAE *= PWE
    Epot = NAE * (1 / 8760) * (1 / 6) * SWF.loc[period].values[0]
    return Epot


def outer_fill(df, period):
    period_dt = dt.strptime(period, "%Y-%m")
    period_dt_upper = period_dt + relativedelta(months=1)
    period_upper = period_dt_upper.strftime("%Y-%m")
    period_dt_lower = period_dt + relativedelta(months=-1)
    period_lower = period_dt_lower.strftime("%Y-%m")

    if df["Parameter"].iat[0] in {"Resumed", "Resumed;Resumed"}:
        first_row = pd.DataFrame(
            {
                "TimeOn": pd.Timestamp(f"{period_lower}-01 00:10:00.000"),
                "Alarmcode": 115,
                "Parameter": ["Stopped"],
                "Was Missing": True,
            }
        )
        df = pd.concat([first_row, df], sort=False).reset_index(drop=True)

    if df["Parameter"].iat[-1] in {"Stopped", "Stopped;Stopped"}:
        last_row = pd.DataFrame(
            {
                "TimeOn": pd.Timestamp(f"{period_upper}-01 00:00:00.000"),
                "Alarmcode": 115,
                "Parameter": ["Resumed"],
                "Was Missing": True,
            }
        )
        df = pd.concat([df, last_row], sort=False).reset_index(drop=True)

    return df


def inner_fill(df, name, alarms_result_sum):

    for j in df.index[:-1]:
        if (df["Parameter"].iat[j] in {"Stopped", "Stopped;Stopped"}) & (
            df["Parameter"].iat[j + 1] in {"Stopped", "Stopped;Stopped"}
        ):

            result_turbine = alarms_result_sum.loc[(alarms_result_sum.StationNr == name)].copy()

            TimeOn = result_turbine.loc[
                (result_turbine.TimeOn.shift(-1) < df["TimeOn"].iat[j + 1])
            ].TimeOff.max()

            if TimeOn is pd.NaT:
                TimeOn = df["TimeOn"].iat[j + 1]

            line = pd.DataFrame(
                {
                    "TimeOn": TimeOn,
                    "Alarmcode": 115,
                    "Parameter": ["Resumed"],
                    "Was Missing": True,
                },
                index=[j + 0.5],
            )

            df = df.append(line, ignore_index=False)
            df = df.sort_index().reset_index(drop=True)

        elif (df["Parameter"].iat[j] in {"Resumed", "Resumed;Resumed"}) & (
            df["Parameter"].iat[j + 1] in {"Resumed", "Resumed;Resumed"}
        ):

            TimeOn = alarms_result_sum.loc[
                (alarms_result_sum.StationNr == name)
                & (alarms_result_sum.TimeOff < df["TimeOn"].iat[j + 1])
                & (alarms_result_sum.TimeOff > df["TimeOn"].iat[j])
                & (alarms_result_sum.TimeOn > df["TimeOn"].iat[j])
            ].TimeOn.min()

            # result_turbine = alarms_result_sum.loc[(
            #     alarms_result_sum.StationNr == name)].copy()

            # TimeOn = result_turbine.loc[(
            #     result_turbine.TimeOff > df['TimeOn'].iat[j])].TimeOn.min()
            if TimeOn is pd.NaT:
                TimeOn = df["TimeOn"].iat[j]

            line = pd.DataFrame(
                {
                    "TimeOn": TimeOn,
                    "Alarmcode": 115,
                    "Parameter": ["Stopped"],
                    "Was Missing": True,
                },
                index=[j + 0.5],
            )
            df = df.append(line, ignore_index=False)
            df = df.sort_index().reset_index(drop=True)

    return df


def fill_115_apply(df, period, name, alarms_result_sum):

    df = df.reset_index(drop=True)

    df = outer_fill(df, period)

    df = inner_fill(df, name, alarms_result_sum)

    df.drop(columns={"StationNr"}, inplace=True)

    return df


def fill_115(alarms, period, alarms_result_sum):

    path = "./monthly_data/115/"

    # load adjusted 115 alamrs if already filled and adjusted
    if (os.path.isfile(path + f"{period}-115-missing.xlsx")) and (
        os.path.isfile(path + f"{period}-115.xlsx")
    ):

        missing_115 = pd.read_excel(path + f"{period}-115-missing.xlsx")
        alarms_115_filled_rows = pd.read_excel(path + f"{period}-115.xlsx")

        alarms_115_filled_rows = alarms_115_filled_rows.loc[
            alarms_115_filled_rows["Was Missing"] != True
        ]

        missing_115 = missing_115.loc[missing_115["Was Missing"] == True]

        alarms_115_filled_rows = (
            alarms_115_filled_rows.append(missing_115, ignore_index=True, sort=False)
            .sort_values(["StationNr", "TimeOn"])
            .reset_index(drop=True)
        )

    # fill 115 alarms if alarms not adjusted
    else:
        alarms_115 = alarms[alarms.Alarmcode == 115].copy()

        alarms_115["Parameter"] = alarms_115["Parameter"].str.replace(" ", "")

        alarms_115 = alarms_115.sort_values(["StationNr", "TimeOn"])

        alarms_115_filled_rows = alarms_115.groupby("StationNr").apply(
            lambda df: fill_115_apply(df, period, df.name, alarms_result_sum)
        )
        alarms_115_filled_rows.reset_index(inplace=True)
        alarms_115_filled_rows = alarms_115_filled_rows.drop("level_1", axis=1)

        missing_115 = alarms_115_filled_rows.groupby("StationNr").apply(
            lambda df: df.loc[
                (
                    (df["Was Missing"] == True)
                    | (df["Was Missing"].shift() == True)
                    | (df["Was Missing"].shift(-1) == True)
                )
            ]
        )

        del missing_115["StationNr"]
        missing_115 = missing_115.reset_index(level=0)

        alarms_115_filled_rows.to_excel(path + f"{period}-115.xlsx")
        missing_115.to_excel(path + f"{period}-115-missing.xlsx")

    return alarms_115_filled_rows


def outer_fill_20(df, period):
    period_dt = dt.strptime(period, "%Y-%m")
    period_dt_upper = period_dt + relativedelta(months=1)
    period_upper = period_dt_upper.strftime("%Y-%m")
    period_dt_lower = period_dt + relativedelta(months=-1)
    period_lower = period_dt_lower.strftime("%Y-%m")

    if df["Parameter"].iat[0] == "Resumed":
        first_row = pd.DataFrame(
            {
                "TimeOn": pd.Timestamp(f"{period_lower}-01 00:10:00.000"),
                "Alarmcode": 25,
                "Parameter": ["Stopped"],
                "Was Missing": True,
            }
        )
        df = pd.concat([first_row, df], sort=False).reset_index(drop=True)

    if df["Parameter"].iat[-1] == "Stopped":
        last_row = pd.DataFrame(
            {
                "TimeOn": pd.Timestamp(f"{period_upper}-01 00:00:00.000"),
                "Alarmcode": 20,
                "Parameter": ["Resumed"],
                "Was Missing": True,
            }
        )
        df = pd.concat([df, last_row], sort=False).reset_index(drop=True)

    return df


def inner_fill_20(df, name, alarms_result_sum):

    for j in df.index[:-1]:
        if (df["Parameter"].iat[j] == "Stopped") & (df["Parameter"].iat[j + 1] == "Stopped"):

            result_turbine = alarms_result_sum.loc[(alarms_result_sum.StationNr == name)].copy()

            TimeOn = result_turbine.loc[
                (result_turbine.TimeOn.shift(-1) < df["TimeOn"].iat[j + 1])
            ].TimeOff.max()

            line = pd.DataFrame(
                {
                    "TimeOn": TimeOn,
                    "Alarmcode": 20,
                    "Parameter": ["Resumed"],
                    "Was Missing": True,
                },
                index=[j + 0.5],
            )

            df = df.append(line, ignore_index=False)
            df = df.sort_index().reset_index(drop=True)

        elif (df["Parameter"].iat[j] == "Resumed") & (df["Parameter"].iat[j + 1] == "Resumed"):

            TimeOn = alarms_result_sum.loc[
                (alarms_result_sum.StationNr == name)
                & (alarms_result_sum.TimeOff < df["TimeOn"].iat[j + 1])
                & (alarms_result_sum.TimeOff > df["TimeOn"].iat[j])
            ].TimeOn.min()

            result_turbine = alarms_result_sum.loc[(alarms_result_sum.StationNr == name)].copy()

            TimeOn = result_turbine.loc[
                (result_turbine.TimeOff > df["TimeOn"].iat[j])
            ].TimeOn.min()

            line = pd.DataFrame(
                {
                    "TimeOn": TimeOn,
                    "Alarmcode": 25,
                    "Parameter": ["Stopped"],
                    "Was Missing": True,
                },
                index=[j + 0.5],
            )
            df = df.append(line, ignore_index=False)
            df = df.sort_index().reset_index(drop=True)

    return df


def fill_20_apply(df, period, name, alarms_result_sum):

    df = df.reset_index(drop=True)

    df = outer_fill_20(df, period)

    df = inner_fill_20(df, name, alarms_result_sum)

    df.drop(columns={"StationNr"}, inplace=True)

    return df


def fill_20(alarms, period, alarms_result_sum):

    path = "./monthly_data/20/"

    # load adjusted 20 alamrs
    if (os.path.isfile(path + f"{period}-20-missing.xlsx")) and (
        os.path.isfile(path + f"{period}-20.xlsx")
    ):

        missing_20 = pd.read_excel(path + f"{period}-cut-missing.xlsx")
        alarms_20_filled_rows = pd.read_excel(path + f"{period}-cut.xlsx")

        alarms_20_filled_rows = alarms_20_filled_rows.loc[
            alarms_20_filled_rows["Was Missing"] != True
        ]

        missing_20 = missing_20.loc[missing_20["Was Missing"] == True]

        alarms_20_filled_rows = (
            alarms_20_filled_rows.append(missing_20, ignore_index=True, sort=False)
            .sort_values(["StationNr", "TimeOn"])
            .reset_index(drop=True)
        )

    # fill 20 alarms
    else:
        alarms_20 = alarms[(alarms.Alarmcode == 20) | (alarms.Alarmcode == 25)].copy()

        alarms_20["Parameter"] = alarms_20["Parameter"].str.replace(" ", "")

        alarms_20 = alarms_20.sort_values(["StationNr", "TimeOn"])

        alarms_20["Parameter"] = alarms_20["Alarmcode"].map({20: "Resumed", 25: "Stopped"})

        alarms_20_filled_rows = alarms_20.groupby("StationNr").apply(
            lambda df: fill_20_apply(df, period, df.name, alarms_result_sum)
        )

        alarms_20_filled_rows.reset_index(inplace=True)
        alarms_20_filled_rows = alarms_20_filled_rows.drop("level_1", axis=1)

        missing_20 = alarms_20_filled_rows.groupby("StationNr").apply(
            lambda df: df.loc[
                (
                    (df["Was Missing"] == True)
                    | (df["Was Missing"].shift() == True)
                    | (df["Was Missing"].shift(-1) == True)
                )
            ]
        )

        del missing_20["StationNr"]
        missing_20 = missing_20.reset_index(level=0)

        alarms_20_filled_rows.to_excel(path + f"{period}-cut.xlsx", index=False)
        missing_20.to_excel(path + f"{period}-cut-missing.xlsx", index=False)
    return alarms_20_filled_rows


def upsample_115_20(df_group, period, alarmcode):

    StationId, df = df_group
    df = df.loc[:, ["TimeOn"]]

    clmn_name = f"Duration {alarmcode}(s)"
    df[clmn_name] = 0

    df = df.set_index("TimeOn")
    df = df.sort_index()

    df.iloc[::2, 0] = 1
    df.iloc[1::2, 0] = -1

    # precision en miliseconds
    precision = 1000

    df = df.resample(f"{precision}ms", label="right").sum().cumsum()

    df.loc[df[clmn_name] > 0, clmn_name] = precision / 1000

    df = df.resample("10T", label="right").sum()

    period_dt = dt.strptime(period, "%Y-%m")
    period_dt_upper = period_dt + relativedelta(months=1)
    period_dt_upper = period_dt_upper.strftime("%Y-%m")

    full_range = pd.date_range(
        pd.Timestamp(f"{period}-01 00:10:00.000"),
        pd.Timestamp(f"{period_dt_upper}-01 00:00:00.000"),
        freq="10T",
    )

    df = df.reindex(index=full_range, fill_value=0)

    df["StationId"] = StationId
    return df


class read_files:

    # ------------------------------grd-------------------------------------
    @staticmethod
    def read_grd(period):
        usecols_grd = """TimeStamp, StationId, wtc_ActPower_min, wtc_ActPower_max,
        wtc_ActPower_mean"""

        sql_grd = f"Select {usecols_grd} FROM tblSCTurGrid;"

        grd = zip_to_df(data_type="grd", sql=sql_grd, period=period)
        grd["TimeStamp"] = pd.to_datetime(grd["TimeStamp"], format="%m/%d/%y %H:%M:%S")

        return grd

    # ------------------------------cnt-------------------------------------
    @staticmethod
    def read_cnt(period):
        usecols_cnt = """TimeStamp, StationId, wtc_kWG1Tot_accum,
        wtc_kWG1TotE_accum"""

        sql_cnt = f"Select {usecols_cnt} FROM tblSCTurCount;"

        cnt = zip_to_df(data_type="cnt", sql=sql_cnt, period=period)

        cnt["TimeStamp"] = pd.to_datetime(cnt["TimeStamp"], format="%m/%d/%y %H:%M:%S")

        return cnt

    # -----------------------------sum---------------------------
    @staticmethod
    def read_sum(period):
        # usecols_sum = """
        # SELECT CDbl(TimeOn) AS TOn, CDbl(TimeOff) AS TOff,
        # StationNr, Alarmcode, ID, Parameter
        # FROM tblAlarmLog WHERE TimeOff IS NOT NULL
        # union
        # SELECT CDbl(TimeOn) AS TOn, TimeOff AS TOff,
        # StationNr, Alarmcode, ID, Parameter
        # FROM tblAlarmLog WHERE TimeOff IS NULL
        # """
        # alarms = zip_to_df("sum", usecols_sum, period)

        # alarms.rename(columns={"TOn": "TimeOn", "TOff": "TimeOff"}, inplace=True)
        # alarms.loc[:, "TimeOn"] = sqldate_to_datetime(alarms["TimeOn"].copy())
        # alarms.loc[:, "TimeOff"] = sqldate_to_datetime(alarms["TimeOff"].copy())

        alarms = pd.read_csv(
            f"./monthly_data/uploads/SUM/{period}-sum.rpt",
            sep="|",
            skipfooter=2
            # on_bad_lines="skip",
        )
        alarms.dropna(subset=["Alarmcode"], inplace=True)

        alarms.loc[:, "TimeOn"] = pd.to_datetime(alarms["TimeOn"], format="%Y-%m-%d %H:%M:%S.%f")
        alarms.loc[:, "TimeOff"] = pd.to_datetime(alarms["TimeOff"], format="%Y-%m-%d %H:%M:%S.%f")

        alarms = alarms[alarms.StationNr >= 2307405]
        alarms = alarms[alarms.StationNr <= 2307535].reset_index(drop=True)
        alarms.reset_index(drop=True, inplace=True)
        alarms["Alarmcode"] = alarms.Alarmcode.astype(int)
        alarms["Parameter"] = alarms.Parameter.str.replace(" ", "")

        return alarms

    # ------------------------------tur---------------------------
    @staticmethod
    def read_tur(period):
        usecols_tur = """TimeStamp, StationId, wtc_AcWindSp_mean, wtc_AcWindSp_stddev,
        wtc_ActualWindDirection_mean, wtc_ActualWindDirection_stddev"""

        sql_tur = f"Select {usecols_tur} FROM tblSCTurbine;"

        tur = zip_to_df("tur", sql_tur, period)

        return tur

    # ------------------------------met---------------------------
    @staticmethod
    def read_met(period):

        usecols_met = """TimeStamp, StationId ,met_WindSpeedRot_mean,
        met_WinddirectionRot_mean"""

        sql_met = f"Select {usecols_met} FROM tblSCMet;"

        met = zip_to_df("met", sql_met, period)

        met = met.pivot(
            "TimeStamp", "StationId", ["met_WindSpeedRot_mean", "met_WinddirectionRot_mean"]
        )

        met.columns = met.columns.to_flat_index()

        met.reset_index(inplace=True)

        met.columns = [
            "_".join(str(v) for v in tup) if type(tup) is tuple else tup for tup in met.columns
        ]

        return met

    @staticmethod
    def read_din(period):
        usecols_din = """TimeStamp, StationId, wtc_PowerRed_timeon"""

        sql_din = f"Select {usecols_din} FROM tblSCTurDigiIn;"

        din = zip_to_df("din", sql_din, period)

        return din

    @staticmethod
    def read_all(period):

        return (
            read_files.read_met(period),
            read_files.read_tur(period),
            read_files.read_sum(period),
            read_files.read_cnt(period),
            read_files.read_grd(period),
            read_files.read_din(period),
        )


def full_calculation(period):

    # reading all files with function
    met, tur, alarms, cnt, grd, din = read_files.read_all(period)

    # cnt = cnt.query("TimeStamp <= '2021-11-29 18:00:00.000'")
    # ------------------------------------------------------------

    period_dt = dt.strptime(period, "%Y-%m")
    period_month = period_dt.month
    previous_period_dt = period_dt + relativedelta(months=-1)
    previous_period = previous_period_dt.strftime("%Y-%m")
    next_period_dt = period_dt + relativedelta(months=1)
    next_period = next_period_dt.strftime("%Y-%m")

    currentMonth = dt.now().month
    currentYear = dt.now().year
    currentPeriod = f"{currentYear}-{str(currentMonth).zfill(2)}"
    currentPeriod_dt = dt.strptime(currentPeriod, "%Y-%m")

    period_start = pd.Timestamp(f"{period}-01 00:00:00.000")

    # if currentPeriod_dt <= period_dt:  # if calculating ongoing month
    period_end = cnt.TimeStamp.max()
    # else:
    #     period_end = pd.Timestamp(f"{next_period}-01 00:10:00.000")

    full_range_var = pd.date_range(period_start, period_end, freq="10T")

    # ----------------------Sanity check---------------------------
    sanity_grd = grd.query(
        """-1000 <= wtc_ActPower_min <= 2600 & -1000 <= wtc_ActPower_max <= 2600 & -1000 <= wtc_ActPower_mean <= 2600"""
    ).index
    sanity_cnt = cnt.query(
        """-500 <= wtc_kWG1Tot_accum <= 500 & 0 <= wtc_kWG1TotE_accum <= 500"""
    ).index
    sanity_tur = tur.query(
        """0 <= wtc_AcWindSp_mean <= 50 & 0 <= wtc_ActualWindDirection_mean <= 360"""
    ).index
    # sanity_met = met.query('''0 <= met_WindSpeedRot_mean <= 50 & 0 <= met_WinddirectionRot_mean <= 360''').index
    sanity_din = din.query("""0 <= wtc_PowerRed_timeon <= 600""").index

    # grd_outliers = grd.loc[grd.index.difference(sanity_grd)]
    # cnt_outliers = cnt.loc[cnt.index.difference(sanity_cnt)].groupby('StationId').apply(
    #     lambda df: df.reindex(index=full_range_var)
    # )

    # cnt_outliers = cnt.groupby('StationId').apply(
    #     lambda df: df.reindex(index=full_range_var))

    # cnt_outliers = cnt_outliers.loc[cnt_outliers.wtc_kWG1TotE_accum.isna()]

    # tur_outliers = tur.loc[tur.index.difference(sanity_tur)]
    # # met_outliers = met.loc[met.index.difference(sanity_met)]
    # din_outliers = din.loc[din.index.difference(sanity_din)]

    # with pd.ExcelWriter(f'./monthly_data/results/outliers/{period}_outliers.xlsx') as writer:
    #     grd_outliers.to_excel(writer, sheet_name='grd')
    #     cnt_outliers.to_excel(writer, sheet_name='cnt')
    #     tur_outliers.to_excel(writer, sheet_name='tur')
    #     din_outliers.to_excel(writer, sheet_name='din')

    grd = grd.loc[grd.index.isin(sanity_grd)]
    cnt = cnt.loc[cnt.index.isin(sanity_cnt)]
    tur = tur.loc[tur.index.isin(sanity_tur)]
    # met = met.loc[met.index.isin(sanity_met)]
    din = din.loc[din.index.isin(sanity_din)]

    # --------------------------error list-------------------------
    error_list = pd.read_excel(r"Error_Type_List_Las_Update_151209.xlsx")

    error_list.Number = error_list.Number.astype(int)  # ,errors='ignore'

    error_list.drop_duplicates(subset=["Number"], inplace=True)

    error_list.rename(columns={"Number": "Alarmcode"}, inplace=True)

    # ------------------------------------------------------

    # for i in range(1, 12):  # append last months alarms

    #     ith_previous_period_dt = period_dt + relativedelta(months=-i)
    #     ith_previous_period = ith_previous_period_dt.strftime("%Y-%m")

    #     try:
    #         previous_alarms = read_files.read_sum(ith_previous_period)
    #         alarms = alarms.append(previous_alarms)

    #     except FileNotFoundError:
    #         print(f"Previous mounth -{i} alarms File not found")

    # ------------------------------------------------------
    alarms_0_1 = error_list.loc[error_list["Error Type"].isin([1, 0])].Alarmcode

    # ------------------------------Fill NA TimeOff-------------------------------------

    alarms["OldTimeOn"] = alarms["TimeOn"]
    alarms["OldTimeOff"] = alarms["TimeOff"]

    print(f"TimeOff NAs = {alarms.loc[alarms.Alarmcode.isin(alarms_0_1)].TimeOff.isna().sum()}")

    if alarms.loc[alarms.Alarmcode.isin(alarms_0_1)].TimeOff.isna().sum():
        print(
            f"earliest TimeOn when TimeOff is NA= \
            {alarms.loc[alarms.Alarmcode.isin(alarms_0_1) & alarms.TimeOff.isna()].TimeOn.min()}"
        )

    alarms.loc[alarms.Alarmcode.isin(alarms_0_1), "TimeOff"] = alarms.loc[
        alarms.Alarmcode.isin(alarms_0_1), "TimeOff"
    ].fillna(period_end)

    # ------------------------------Alarms ending after period end ----------------------

    alarms.loc[(alarms.TimeOff > period_end), "TimeOff"] = period_end

    # ------------------------------Keep only alarms active in period-------------
    alarms.reset_index(inplace=True, drop=True)
    # ----dropping 1 0 alarms
    alarms.drop(
        alarms.query(
            "(TimeOn < @period_start) & (TimeOff < @period_start) & Alarmcode.isin(@alarms_0_1)"
        ).index,
        inplace=True,
    )

    alarms.drop(alarms.query("(TimeOn > @period_end)").index, inplace=True)
    alarms.reset_index(drop=True, inplace=True)
    # ------------------------------Alarms starting before period start -----------------
    warning_date = alarms.TimeOn.min()

    alarms.loc[
        (alarms.TimeOn < period_start) & (alarms.Alarmcode.isin(alarms_0_1)), "TimeOn"
    ] = period_start

    # ----dropping non 1 0 alarms
    alarms.drop(
        alarms.query("~Alarmcode.isin(@alarms_0_1) & (TimeOn < @period_start)").index,
        inplace=True,
    )
    alarms.reset_index(drop=True, inplace=True)

    """ label scada alarms with coresponding error type
    and only keep alarm codes in error list"""
    result_sum = pd.merge(alarms, error_list, on="Alarmcode", how="inner", sort=False)

    # Remove warnings
    result_sum = result_sum.loc[result_sum["Error Type"].isin([1, 0])]

    # Determine alarms real periods applying cascade method

    # apply cascade
    alarms_result_sum = apply_cascade(result_sum)
    alarms_result_sum = remove_1005_overlap(alarms_result_sum)

    # ----------------openning pool for multiprocessing------------------------
    pool = mp.Pool(processes=(mp.cpu_count() - 0))

    # -------------------2006  binning --------------------------------------

    print("binning 2006")

    alarms_df_2006 = alarms.loc[(alarms["Alarmcode"] == 2006)].copy()
    # alarms_df_2006['TimeOff'] = alarms_df_2006['NewTimeOn']
    alarms_df_2006["NewTimeOn"] = alarms_df_2006["TimeOn"]

    alarms_df_2006 = alarms_df_2006.query(
        "(@period_start < TimeOn < @period_end) | \
                                   (@period_start < TimeOff < @period_end) | \
                                   ((TimeOn < @period_start) & (@period_end < TimeOff))"
    )

    if not alarms_df_2006.empty:
        alarms_df_2006_10min = realperiod_10mins(alarms_df_2006, "2006")

        alarms_df_2006_10min = alarms_df_2006_10min.groupby("TimeStamp").agg(
            {"RealPeriod": "sum", "StationNr": "first"}
        )

        alarms_df_2006_10min.reset_index(inplace=True)

        alarms_df_2006_10min.rename(
            columns={"StationNr": "StationId", "RealPeriod": "Duration 2006(s)"}, inplace=True
        )

        alarms_df_2006_10min["Duration 2006(s)"] = alarms_df_2006_10min[
            "Duration 2006(s)"
        ].dt.total_seconds()

    else:
        print("no 2006")
        alarms_df_2006_10min = pd.DataFrame(columns={"TimeStamp", "Duration 2006(s)", "StationId"})
    # ------------------115 filling binning -----------------------------------

    print("filling 115")
    alarms_115_filled = fill_115(
        alarms.query("@period_start < TimeOn < @period_end"), period, alarms_result_sum
    )
    print("115 filled")

    print("upsampling 115")
    grp_lst_args = iter([(n, period, "115") for n in alarms_115_filled.groupby("StationNr")])

    alarms_115_filled_binned = pool.starmap(upsample_115_20, grp_lst_args)

    alarms_115_filled_binned = pd.concat(alarms_115_filled_binned)

    alarms_115_filled_binned.reset_index(inplace=True)

    alarms_115_filled_binned.rename(columns={"index": "TimeStamp"}, inplace=True)

    # -------------------20/25 filling binning --------------------------------

    print("filling 20")
    alarms_20_filled = fill_20(
        alarms.query("@period_start < TimeOn < @period_end"), period, alarms_result_sum
    )
    print("20 filled")
    grp_lst_args = iter([(n, period, "20-25") for n in alarms_20_filled.groupby("StationNr")])

    print("upsampling 20-25")
    alarms_20_filled_binned = pool.starmap(upsample_115_20, grp_lst_args)

    alarms_20_filled_binned = pd.concat(alarms_20_filled_binned)

    alarms_20_filled_binned.reset_index(inplace=True)

    alarms_20_filled_binned.rename(columns={"index": "TimeStamp"}, inplace=True)

    # # Remove previous period alarms
    # mask = (
    #     (alarms_result_sum["TimeOn"].dt.month == period_month)
    #     | (alarms_result_sum["TimeOff"].dt.month == period_month)
    #     | (alarms_result_sum["TimeOff"] == period_end)
    # )

    # alarms_result_sum = alarms_result_sum.loc[mask]
    pool.close()

    # ----------------------- binning --------------------------------------

    print("Binning")
    # Binning alarms and remove overlap with 1005 (new method)

    alarms_df_clean = alarms_result_sum.loc[
        (alarms_result_sum["RealPeriod"].dt.total_seconds() != 0)
    ].copy()

    alarms_df_clean_10min = realperiod_10mins(alarms_df_clean)
    alarms_df_clean_10min.reset_index(inplace=True, drop=True)

    # # ----------------------- ---------------------------------

    alarms_binned = (
        alarms_df_clean_10min.groupby(["StationNr", "TimeStamp"])
        .agg(
            {
                "RealPeriod": "sum",
                "Period Tarec(s)": "sum",
                "Period Siemens(s)": "sum",
                "UK Text": "|".join,
            }
        )
        .reset_index()
    )

    alarms_binned = (
        alarms_binned.groupby("StationNr")
        .apply(lambda df: full_range(df, full_range_var))
        .reset_index()
        .rename(
            columns={
                "level_1": "TimeStamp",
                "StationNr": "StationId",
                "Period Tarec(s)": "Period 0(s)",
                "Period Siemens(s)": "Period 1(s)",
            }
        )
    )

    alarms_binned["Period 0(s)"] = alarms_binned["Period 0(s)"].dt.total_seconds().fillna(0)
    alarms_binned["Period 1(s)"] = alarms_binned["Period 1(s)"].dt.total_seconds().fillna(0)
    alarms_binned["RealPeriod"] = alarms_binned["RealPeriod"].dt.total_seconds().fillna(0)

    alarms_binned.drop(
        alarms_binned.loc[alarms_binned["TimeStamp"] == period_start].index, inplace=True
    )

    print("Alarms Binned")

    # ----------Merging with 2006 alarms

    alarms_binned = pd.merge(
        alarms_binned, alarms_df_2006_10min, on=["TimeStamp", "StationId"], how="left"
    ).reset_index(drop=True)

    # -------merging cnt, grd, tur, met,upsampled (alarms ,115 and 20/25)------
    # merging upsampled alarms with energy production

    print("merging upsampled alarms with energy production")
    cnt_alarms = pd.merge(
        alarms_binned, cnt, on=["TimeStamp", "StationId"], how="left"
    ).reset_index(drop=True)

    # merging last dataframe with power
    cnt_alarms_minpwr = pd.merge(cnt_alarms, grd, on=["TimeStamp", "StationId"], how="left")

    cnt_alarms_minpwr.reset_index(drop=True, inplace=True)

    # merging last dataframe with 115 upsampled
    cnt_115 = pd.merge(
        cnt_alarms_minpwr, alarms_115_filled_binned, on=["StationId", "TimeStamp"], how="left"
    )

    # merging last dataframe with 20/25 upsampled
    cnt_115 = pd.merge(cnt_115, alarms_20_filled_binned, on=["StationId", "TimeStamp"], how="left")

    # merging last dataframe with turbine windspeed data
    cnt_115 = pd.merge(
        cnt_115,
        tur[["TimeStamp", "StationId", "wtc_AcWindSp_mean", "wtc_ActualWindDirection_mean"]],
        on=("TimeStamp", "StationId"),
        how="left",
    )

    # merging last dataframe with met mast data
    cnt_115 = pd.merge(cnt_115, met, on="TimeStamp", how="left")

    # merging last dataframe with curtailement
    cnt_115 = pd.merge(cnt_115, din, on=["StationId", "TimeStamp"], how="left")

    cnt_115 = cnt_115.fillna(0)

    # -------- operational turbines mask --------------------------------------
    mask_n = (
        (cnt_115["wtc_kWG1TotE_accum"] > 0)
        & (cnt_115["Period 0(s)"] == 0)
        & (cnt_115["Period 1(s)"] == 0)
        & (cnt_115["wtc_ActPower_min"] > 0)
        & ((cnt_115["Duration 115(s)"] == 0) | (cnt_115["wtc_ActPower_min"] > 0))
        & ((cnt_115["Duration 20-25(s)"] == 0) | (cnt_115["wtc_ActPower_min"] > 0))
        & (cnt_115["Duration 2006(s)"] == 0)
        & (cnt_115["wtc_PowerRed_timeon"] == 0)
    )

    # -------- operational turbines -------------------------------------------
    cnt_115_n = cnt_115.loc[mask_n].copy()

    cnt_115_n["Correction Factor"] = 0
    cnt_115_n["Available Turbines"] = 0

    Epot = (
        cnt_115_n.groupby("TimeStamp")
        .agg(
            {
                "wtc_kWG1TotE_accum": ep_cf,
                "Correction Factor": cf_column,
                "Available Turbines": "count",
            }
        )
        .copy()
    )

    Epot = Epot.rename(columns={"wtc_kWG1TotE_accum": "Epot"})

    del cnt_115_n["Correction Factor"]
    del cnt_115_n["Available Turbines"]

    cnt_115_n = pd.merge(cnt_115_n, Epot, on="TimeStamp", how="left")

    cnt_115_n["Epot"] = cnt_115_n["wtc_kWG1TotE_accum"]

    cnt_115_no = cnt_115.loc[~mask_n].copy()

    cnt_115_no = pd.merge(cnt_115_no, Epot, on="TimeStamp", how="left")

    mask = cnt_115_no["Epot"] < cnt_115_no["wtc_kWG1TotE_accum"]

    cnt_115_no.loc[mask, "Epot"] = cnt_115_no.loc[mask, "wtc_kWG1TotE_accum"]

    cnt_115_final = pd.DataFrame()

    cnt_115_final = cnt_115_no.append(cnt_115_n, sort=False)

    cnt_115_final = cnt_115_final.sort_values(["StationId", "TimeStamp"]).reset_index(drop=True)

    # mask_Epot_case_2 = cnt_115_final["Epot"].isna()

    # Epot_case_2_var = Epot_case_2(cnt_115_final.loc[mask_Epot_case_2])

    # cnt_115_final.loc[mask_Epot_case_2, "Epot"] = np.maximum(
    #     Epot_case_2_var, cnt_115_final.loc[mask_Epot_case_2, "wtc_kWG1TotE_accum"].values
    # )
    cnt_115_final["EL"] = cnt_115_final["Epot"].fillna(0) - cnt_115_final[
        "wtc_kWG1TotE_accum"
    ].fillna(0)

    cnt_115_final["EL"] = cnt_115_final["EL"].clip(lower=0)

    cnt_115_final = cnt_115_final.fillna(0)

    cnt_115_final["ELX"] = (
        (
            cnt_115_final["Period 0(s)"]
            / (cnt_115_final["Period 0(s)"] + cnt_115_final["Period 1(s)"])
        )
        * (cnt_115_final["EL"])
    ).fillna(0)

    cnt_115_final["ELNX"] = (
        (
            cnt_115_final["Period 1(s)"]
            / (cnt_115_final["Period 0(s)"] + cnt_115_final["Period 1(s)"])
        )
        * (cnt_115_final["EL"])
    ).fillna(0)

    cnt_115_final.loc[cnt_115_final["Duration 115(s)"] > 0, "EL 115"] = cnt_115_final.loc[
        cnt_115_final["Duration 115(s)"] > 0, "EL"
    ]

    cnt_115_final["EL 115"] = cnt_115_final["EL 115"].fillna(0)

    cnt_115_final.loc[cnt_115_final["Duration 20-25(s)"] > 0, "EL 20-25"] = cnt_115_final.loc[
        cnt_115_final["Duration 20-25(s)"] > 0, "EL"
    ]

    cnt_115_final["EL 20-25"] = cnt_115_final["EL 20-25"].fillna(0)

    cnt_115_final["EL_115_left"] = cnt_115_final["EL 115"] - (
        cnt_115_final["ELX"] + cnt_115_final["ELNX"]
    )

    max_115_ELX_ELNX = pd.concat(
        [(cnt_115_final["ELX"] + cnt_115_final["ELNX"]), cnt_115_final["EL 115"]], axis=1
    ).max(axis=1)

    cnt_115_final["EL_indefini"] = cnt_115_final["EL"] - max_115_ELX_ELNX

    # -------------------------------------------------------------------------

    cnt_115_final["prev_AcWindSp"] = cnt_115_final.groupby("TimeStamp")[
        "wtc_AcWindSp_mean"
    ].shift()

    cnt_115_final["next_AcWindSp"] = cnt_115_final.groupby("TimeStamp")["wtc_AcWindSp_mean"].shift(
        -1
    )

    cnt_115_final["prev_ActPower_min"] = cnt_115_final.groupby("TimeStamp")[
        "wtc_ActPower_min"
    ].shift()

    cnt_115_final["next_ActPower_min"] = cnt_115_final.groupby("TimeStamp")[
        "wtc_ActPower_min"
    ].shift(-1)

    cnt_115_final["prev_Alarme"] = cnt_115_final.groupby("TimeStamp")["RealPeriod"].shift()

    cnt_115_final["next_Alarme"] = cnt_115_final.groupby("TimeStamp")["RealPeriod"].shift(-1)

    cnt_115_final["DiffV1"] = cnt_115_final.prev_AcWindSp - cnt_115_final.wtc_AcWindSp_mean

    cnt_115_final["DiffV2"] = cnt_115_final.next_AcWindSp - cnt_115_final.wtc_AcWindSp_mean

    # -------------------------------------------------------------------------

    mask_4 = (cnt_115_final["EL_indefini"] > 0) & (
        cnt_115_final["wtc_PowerRed_timeon"] > 0
    )  # & warning 2006 > 0

    cnt_115_final.loc[mask_4, "EL_PowerRed"] = cnt_115_final.loc[mask_4, "EL_indefini"]

    cnt_115_final["EL_PowerRed"] = cnt_115_final["EL_PowerRed"].fillna(0)

    cnt_115_final["EL_indefini"] = (
        cnt_115_final["EL_indefini"].fillna(0) - cnt_115_final["EL_PowerRed"]
    )

    cnt_115_final["EL_PowerRed"].fillna(0, inplace=True)

    # -------------------------------------------------------------------------

    mask_5 = (cnt_115_final["EL_indefini"] > 0) & (
        cnt_115_final["Duration 2006(s)"] > 0
    )  # & warning 2006 > 0

    cnt_115_final.loc[mask_5, "EL_2006"] = cnt_115_final.loc[mask_5, "EL_indefini"]

    cnt_115_final["EL_2006"] = cnt_115_final["EL_2006"].fillna(0)

    cnt_115_final["EL_indefini"] = (
        cnt_115_final["EL_indefini"].fillna(0) - cnt_115_final["EL_2006"]
    )

    # -------------------------------------------------------------------------

    def lowind(df):

        etape1 = (
            (df.DiffV1 > 1)
            & (df.DiffV2 > 1)
            & ((df.prev_AcWindSp >= 5) | (df.next_AcWindSp >= 5) | (df.wtc_AcWindSp_mean >= 5))
        )

        mask_1 = ~(
            etape1
            & (
                ((df.prev_ActPower_min > 0) & (df.next_ActPower_min > 0))
                | ((df.prev_ActPower_min > 0) & (df.next_Alarme > 0))
                | ((df.next_ActPower_min > 0) & (df.prev_Alarme > 0))
            )
        ) & (df["EL_indefini"] > 0)

        mask_2 = mask_1.shift().bfill()

        df.loc[mask_1, "EL_wind"] = df.loc[mask_1, "EL_indefini"].fillna(0)

        df.loc[mask_1, "Duration lowind(s)"] = 600

        df.loc[mask_2 & ~mask_1, "EL_wind_start"] = (
            df.loc[mask_2 & ~mask_1, "EL_indefini"]
        ).fillna(0)

        df.loc[mask_2 & ~mask_1, "Duration lowind_start(s)"] = 600
        # ---------------------------------------------------------------------

        mask_3 = (
            (df["Duration 115(s)"] > 0).shift()
            & (df["EL_indefini"] > 0)
            & (df["Duration 115(s)"] == 0)
        )

        df.loc[~mask_1 & ~mask_2 & mask_3, "EL_alarm_start"] = (
            df.loc[~mask_1 & ~mask_2 & mask_3, "EL_indefini"]
        ).fillna(0)

        df.loc[~mask_1 & ~mask_2 & mask_3, "Duration alarm_start(s)"] = 600

        return df

    cnt_115_final = cnt_115_final.groupby("StationId").apply(lowind)

    cnt_115_final["EL_indefini_left"] = cnt_115_final["EL_indefini"].fillna(0) - (
        cnt_115_final["EL_wind"].fillna(0)
        + cnt_115_final["EL_wind_start"].fillna(0)
        + cnt_115_final["EL_alarm_start"].fillna(0)
    )

    # -------------------------------------------------------------------------
    # # bypass -------------
    # print('bypassing')

    # cnt_115_final = pd.read_csv(
    #     f"./monthly_data/results/{period}-Availability.csv",
    #     decimal=',', sep=';')

    # cnt_115_final['EL_Misassigned'] = 0

    # # #----end bypass

    # ---------Misassigned low wind---------------
    EL_Misassigned_mask = (
        cnt_115_final["UK Text"].str.contains("low wind")
        & (cnt_115_final["DiffV1"] > 1)
        & (cnt_115_final["DiffV2"] > 1)
        & (
            (cnt_115_final["prev_AcWindSp"] >= 5)
            | (cnt_115_final["next_AcWindSp"] >= 5)
            | (cnt_115_final["wtc_AcWindSp_mean"] >= 5)
        )
    )

    cnt_115_final.loc[EL_Misassigned_mask, "EL_Misassigned"] = cnt_115_final.loc[
        EL_Misassigned_mask, "ELX"
    ]

    cnt_115_final["EL_Misassigned"].fillna(0, inplace=True)
    # -------------------------------------------------------------------------

    columns_toround = list(set(cnt_115_final.columns) - set(("StationId", "TimeStamp", "UK Text")))
    cnt_115_final[columns_toround] = cnt_115_final[columns_toround].round(2).astype(np.float32)

    # -------------------------------------------------------------------------
    print(f"warning: first date in alarm = {warning_date}")

    cnt_115_final.drop(
        cnt_115_final.loc[cnt_115_final["TimeStamp"] == period_start].index, inplace=True
    )

    return cnt_115_final


if __name__ == "__main__":

    full_calculation("2022-07")
