from dateutil.relativedelta import relativedelta
from datetime import datetime as dt
import pyodbc
from zipfile import ZipFile
import os
import numpy as np
import multiprocessing as mp
import pandas as pd


def zip_to_df(data_type, sql, period):

    file_name = f'{period}-{data_type}'
    print(f'Extracting {file_name}')

    ZipFile(f'./monthly_data/uploads/{period}/{file_name}.zip',
            'r').extractall(f'./monthly_data/uploads/{period}/')

    print(f'{data_type} Extracted - Loading')

    conn_str = (
        r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
        fr'DBQ=.\monthly_data\uploads\{period}\{file_name}.mdb;'
    )
    print(conn_str)
    cnxn = pyodbc.connect(conn_str)

    df = pd.read_sql(sql, cnxn)

    print(f'{data_type} Loaded')
    return df


def sqldate_to_datetime(column):
    try:
        column = column.str.replace(',', '.').astype(float)
    except:
        pass
    day_parts = np.modf(column.loc[~column.isna()])

    column.loc[~column.isna()] = (
        dt(1899, 12, 30) +
        day_parts[1].astype('timedelta64[D]', errors='ignore') +
        (day_parts[0] * 86400000).astype('timedelta64[ms]', errors='ignore')
    )
    column = column.fillna(pd.NaT)
    return column


# Determine alarms real periods
def cascade(df):

    df.reset_index(inplace=True, drop=True)
    df['TimeOffMax'] = df.TimeOff.cummax().shift()

    df.at[0, 'TimeOffMax'] = df.at[0, 'TimeOn']

    return df


# looping through turbines and applying cascade method
def apply_cascade(result_sum):

    # Sort by alarm ID
    result_sum.sort_values(['ID'], inplace=True)
    df = result_sum.groupby('StationNr').apply(cascade)

    mask_root = (df.TimeOn.values >= df.TimeOffMax.values)
    mask_children = (df.TimeOn.values < df.TimeOffMax.values) & (
        df.TimeOff.values > df.TimeOffMax.values)
    mask_embedded = (df.TimeOff.values <= df.TimeOffMax.values)

    df.loc[mask_root, 'NewTimeOn'] = df.loc[mask_root, 'TimeOn']
    df.loc[mask_children, 'NewTimeOn'] = df.loc[mask_children, 'TimeOffMax']
    df.loc[mask_embedded, 'NewTimeOn'] = df.loc[mask_embedded, 'TimeOff']

    df.drop(columns=['TimeOffMax'], inplace=True)

    df.reset_index(inplace=True, drop=True)

    TimeOff = df.TimeOff
    NewTimeOn = df.NewTimeOn

    df['RealPeriod'] = abs(TimeOff - NewTimeOn)

    mask_siemens = (df['Error Type'] == 1)
    mask_tarec = (df['Error Type'] == 0)

    df['RealPeriodSiemens(s)'] = df[mask_siemens].RealPeriod  # .dt.seconds
    df['RealPeriodTarec(s)'] = df[mask_tarec].RealPeriod  # .dt.seconds

    df = df.rename(
        columns={"RealPeriodSiemens(s)": "Period Siemens(s)",
                 "RealPeriodTarec(s)": "Period Tarec(s)"})

    return df


def upsample(df_group, period):

    StationId, df = df_group

    df = df.loc[:, ['NewTimeOn', 'TimeOff',
                    'Period Siemens(s)', 'Period Tarec(s)']]

    df.loc[:, ['Period Siemens(s)', 'Period Tarec(s)']] = df[[
        'Period Siemens(s)', 'Period Tarec(s)']].fillna(pd.Timedelta(0))

    result = pd.melt(df,
                     id_vars=['Period Tarec(s)', 'Period Siemens(s)'],
                     value_name='date')

    result['Period 0(s)'] = result['variable'].map(
        {'NewTimeOn': 1, 'TimeOff': -1})

    result['Period 1(s)'] = result['Period 0(s)']

    result['Period 0(s)'] *= result[
        'Period Tarec(s)'].dt.total_seconds().fillna(0).astype(int)

    result['Period 1(s)'] *= result[
        'Period Siemens(s)'].dt.total_seconds().fillna(0).astype(int)

    result = result.set_index('date')

    result = result.sort_index()

    # precision en miliseconds
    precision = 1000

    result = result.resample(f'{precision}ms', label='right').sum().cumsum()

    result.loc[result['Period 0(s)'] > 0, 'Period 0(s)'] = precision / 1000

    result.loc[result['Period 1(s)'] > 0, 'Period 1(s)'] = precision / 1000

    result = result.resample('10T', label='right').sum()

    period_dt = dt.strptime(period, '%Y-%m')

    period_dt_upper = period_dt + relativedelta(months=1)

    period_dt_upper = period_dt_upper.strftime('%Y-%m')

    full_range = pd.date_range(pd.Timestamp(f'{period}-01 00:10:00.000'),
                               pd.Timestamp(
                                   f'{period_dt_upper}-01 00:00:00.000'),
                               freq='10T')

    result = result.reindex(index=full_range, fill_value=0)

    result['StationId'] = StationId

    return result


def alarms_to_10min(alarms_df_clean_tur, period, next_period):

    last_df = pd.DataFrame()
    for i, j in alarms_df_clean_tur.iterrows():
        full_range = pd.date_range(pd.Timestamp(f'{period}-01 00:10:00.000'),
                                   pd.Timestamp(
                                       f'{next_period}-01 00:00:00.000'),
                                   freq='10T')
        df = pd.DataFrame(index=full_range)
        new_j = pd.DataFrame(j).T
        df = df.loc[(df.index >= j.NewTimeOn) & (
            df.index <= (j.TimeOff + pd.Timedelta(minutes=10)))]
        df = pd.concat([df, new_j]).bfill()
        df.drop(df.tail(1).index, inplace=True)
        last_df = pd.concat([last_df, df])

    return last_df[['TimeOn', 'TimeOff', 'Alarmcode',
                    'UK Text', 'Error Type', 'NewTimeOn']]

    # -------------------------------------------------------------------------


def realperiod_10mins(last_df):
    last_df['RealPeriod'] = pd.Timedelta(0)
    last_df['Period Siemens(s)'] = pd.Timedelta(0)
    last_df['Period Tarec(s)'] = pd.Timedelta(0)

    mask_siemens = last_df['Error Type'] == 1
    mask_tarec = last_df['Error Type'] == 0

    df_TimeOn = last_df['NewTimeOn'].reset_index()
    df_TimeOff = last_df['TimeOff'].reset_index()

    df_TimeOn['level_1'] = df_TimeOn['level_1'] - pd.Timedelta(minutes=10)

    last_df['10minTimeOn'] = df_TimeOn[['level_1', 'NewTimeOn']].max(1).values

    last_df['10minTimeOff'] = df_TimeOff[['level_1', 'TimeOff']].min(1).values

    last_df['RealPeriod'] = last_df['10minTimeOff'] - last_df['10minTimeOn']

    last_df.loc[mask_siemens, 'Period Siemens(s)'] = (
        last_df.loc[mask_siemens, 'RealPeriod'])
    last_df.loc[mask_tarec, 'Period Tarec(s)'] = (
        last_df.loc[mask_tarec, 'RealPeriod'])

    return last_df


def remove_1005_overlap(merged_last_df, StationNr, alarms_df_1005_10min):
    # if len(merged_last_df) <= 1:
    #     return merged_last_df
    # print(merged_last_df)
    for i, j in merged_last_df.iterrows():
        if j['Alarmcode'] != 1005:
            df = alarms_df_1005_10min.loc[
                alarms_df_1005_10min['StationNr'] == StationNr]
            if df.empty:
                break
            df['j_10minTimeOn'] = j['10minTimeOn']
            df['j_10minTimeOff'] = j['10minTimeOff']
            df['Timedelta(0)'] = pd.Timedelta(0)
            latest_start = df[['10minTimeOn', 'j_10minTimeOn']].max(1)
            earliest_end = df[['10minTimeOff', 'j_10minTimeOff']].min(1)

            df['delta'] = (earliest_end - latest_start)

            df['Overlap'] = df[['delta', 'Timedelta(0)']].max(1)

            merged_last_df.loc[i, 'RealPeriod'] = (
                merged_last_df.loc[i, 'RealPeriod'] - df['Overlap'].sum())
    return merged_last_df


def full_range(df, period, next_period):
    month_range = pd.date_range(pd.Timestamp(f'{period}-01 00:10:00.000'),
                                pd.Timestamp(f'{next_period}-01 00:00:00.000'),
                                freq='10T')

    new_df = pd.DataFrame(index=month_range)

    df = df.set_index('TimeStamp')
    df = df.drop('StationNr', axis=1)

    return new_df.join(df, how='left')


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


def outer_fill(df, period):
    period_dt = dt.strptime(period, '%Y-%m')
    period_dt_upper = period_dt + relativedelta(months=1)
    period_upper = period_dt_upper.strftime('%Y-%m')
    period_dt_lower = period_dt + relativedelta(months=-1)
    period_lower = period_dt_lower.strftime('%Y-%m')

    if df['Parameter'].iat[0] == 'Resumed':
        first_row = pd.DataFrame({
            'TimeOn': pd.Timestamp(f'{period_lower}-01 00:10:00.000'),
            'Alarmcode': 115,
            'Parameter': ['Stopped'],
            'Was Missing': True})
        df = pd.concat([first_row, df], sort=False).reset_index(
            drop=True)

    if df['Parameter'].iat[-1] == 'Stopped':
        last_row = pd.DataFrame({
            'TimeOn': pd.Timestamp(f'{period_upper}-01 00:00:00.000'),
            'Alarmcode': 115,
            'Parameter': ['Resumed'],
            'Was Missing': True})
        df = pd.concat([df, last_row], sort=False).reset_index(
            drop=True)

    return df


def inner_fill(df, name, alarms_result_sum):

    for j in df.index[:-1]:
        if (df['Parameter'].iat[j] == 'Stopped') & (
                df['Parameter'].iat[j + 1] == 'Stopped'):

            result_turbine = alarms_result_sum.loc[(
                alarms_result_sum.StationNr == name)].copy()

            TimeOn = (result_turbine.loc[(
                result_turbine.TimeOn.shift(-1) < df['TimeOn'].iat[j + 1])]
                .TimeOff.max())

            line = pd.DataFrame(
                {'TimeOn': TimeOn, 'Alarmcode': 115,
                 'Parameter': ['Resumed'],
                 'Was Missing': True}, index=[j + 0.5])

            df = df.append(line, ignore_index=False)
            df = df.sort_index().reset_index(drop=True)

        elif (df['Parameter'].iat[j] == 'Resumed') & (
                df['Parameter'].iat[j + 1] == 'Resumed'):

            TimeOn = alarms_result_sum.loc[(
                alarms_result_sum.StationNr == name) & (
                alarms_result_sum.TimeOff < df['TimeOn'].iat[j + 1]) & (
                alarms_result_sum.TimeOff > df['TimeOn'].iat[j])].TimeOn.min()

            result_turbine = alarms_result_sum.loc[(
                alarms_result_sum.StationNr == name)].copy()

            TimeOn = result_turbine.loc[(
                result_turbine.TimeOff > df['TimeOn'].iat[j])].TimeOn.min()

            line = pd.DataFrame(
                {'TimeOn': TimeOn, 'Alarmcode': 115,
                 'Parameter': ['Stopped'],
                 'Was Missing': True}, index=[j + 0.5])
            df = df.append(line, ignore_index=False)
            df = df.sort_index().reset_index(drop=True)

    return df


def fill_115_apply(df, period, name, alarms_result_sum):

    df = df.reset_index(drop=True)

    df = outer_fill(df, period)

    df = inner_fill(df, name, alarms_result_sum)

    df.drop(columns={'StationNr'}, inplace=True)

    return df


def fill_115(alarms, period, alarms_result_sum):

    path = './monthly_data/115/'

    # load adjusted 115 alamrs if already filled and adjusted
    if (os.path.isfile(path + f"{period}-115-missing.xlsx")) and (
            os.path.isfile(path + f"{period}-115.xlsx")):

        missing_115 = pd.read_excel(path + f"{period}-115-missing.xlsx")
        alarms_115_filled_rows = pd.read_excel(path + f"{period}-115.xlsx")

        alarms_115_filled_rows = alarms_115_filled_rows.loc[
            alarms_115_filled_rows['Was Missing'] != True]

        missing_115 = missing_115.loc[
            missing_115['Was Missing'] == True]

        alarms_115_filled_rows = alarms_115_filled_rows.append(
            missing_115,
            ignore_index=True,
            sort=False).sort_values(
                ['StationNr', 'TimeOn']).reset_index(drop=True)

    # fill 115 alarms if alarms not adjusted
    else:
        alarms_115 = alarms[alarms.Alarmcode == 115].copy()

        alarms_115['Parameter'] = alarms_115[
            'Parameter'].str.replace(' ', '')

        alarms_115 = alarms_115.sort_values(['StationNr', 'TimeOn'])

        alarms_115_filled_rows = alarms_115.groupby(
            'StationNr').apply(lambda df: fill_115_apply(df,
                                                         period,
                                                         df.name,
                                                         alarms_result_sum))
        alarms_115_filled_rows.reset_index(inplace=True)
        alarms_115_filled_rows = alarms_115_filled_rows.drop(
            'level_1', axis=1)

        missing_115 = alarms_115_filled_rows.groupby('StationNr').apply(
            lambda df: df.loc[((
                df['Was Missing'] == True) | (
                df['Was Missing'].shift() == True) | (
                df['Was Missing'].shift(-1) == True))])

        del missing_115['StationNr']
        missing_115 = missing_115.reset_index(level=0)

        alarms_115_filled_rows.to_excel(path + f"{period}-115.xlsx")
        missing_115.to_excel(path + f"{period}-115-missing.xlsx")

    return alarms_115_filled_rows


def outer_fill_20(df, period):
    period_dt = dt.strptime(period, '%Y-%m')
    period_dt_upper = period_dt + relativedelta(months=1)
    period_upper = period_dt_upper.strftime('%Y-%m')
    period_dt_lower = period_dt + relativedelta(months=-1)
    period_lower = period_dt_lower.strftime('%Y-%m')

    if df['Parameter'].iat[0] == 'Resumed':
        first_row = pd.DataFrame(
            {'TimeOn': pd.Timestamp(f'{period_lower}-01 00:10:00.000'),
             'Alarmcode': 25,
             'Parameter': ['Stopped'],
                'Was Missing': True})
        df = pd.concat([first_row, df], sort=False).reset_index(
            drop=True)

    if df['Parameter'].iat[-1] == 'Stopped':
        last_row = pd.DataFrame({
            'TimeOn': pd.Timestamp(f'{period_upper}-01 00:00:00.000'),
            'Alarmcode': 20,
            'Parameter': ['Resumed'],
            'Was Missing': True})
        df = pd.concat([df, last_row], sort=False).reset_index(
            drop=True)

    return df


def inner_fill_20(df, name, alarms_result_sum):

    for j in df.index[:-1]:
        if (df['Parameter'].iat[j] == 'Stopped') & (
                df['Parameter'].iat[j + 1] == 'Stopped'):

            result_turbine = alarms_result_sum.loc[(
                alarms_result_sum.StationNr == name)].copy()

            TimeOn = (result_turbine.loc[(
                result_turbine.TimeOn.shift(-1) < df['TimeOn'].iat[j + 1])]
                .TimeOff.max())

            line = pd.DataFrame(
                {'TimeOn': TimeOn, 'Alarmcode': 20,
                 'Parameter': ['Resumed'],
                 'Was Missing': True}, index=[j + 0.5])

            df = df.append(line, ignore_index=False)
            df = df.sort_index().reset_index(drop=True)

        elif (df['Parameter'].iat[j] == 'Resumed') & (
                df['Parameter'].iat[j + 1] == 'Resumed'):

            TimeOn = alarms_result_sum.loc[(
                alarms_result_sum.StationNr == name) & (
                alarms_result_sum.TimeOff < df['TimeOn'].iat[j + 1]) & (
                alarms_result_sum.TimeOff > df['TimeOn'].iat[j])].TimeOn.min()

            result_turbine = alarms_result_sum.loc[(
                alarms_result_sum.StationNr == name)].copy()

            TimeOn = result_turbine.loc[(
                result_turbine.TimeOff > df['TimeOn'].iat[j])].TimeOn.min()

            line = pd.DataFrame(
                {'TimeOn': TimeOn, 'Alarmcode': 25,
                 'Parameter': ['Stopped'],
                 'Was Missing': True}, index=[j + 0.5])
            df = df.append(line, ignore_index=False)
            df = df.sort_index().reset_index(drop=True)

    return df


def fill_20_apply(df, period, name, alarms_result_sum):

    df = df.reset_index(drop=True)

    df = outer_fill_20(df, period)

    df = inner_fill_20(df, name, alarms_result_sum)

    df.drop(columns={'StationNr'}, inplace=True)

    return df


def fill_20(alarms, period, alarms_result_sum):

    path = './monthly_data/20/'

    # load adjusted 20 alamrs
    if (os.path.isfile(path + f"{period}-20-missing.xlsx")) and (
            os.path.isfile(path + f"{period}-20.xlsx")):

        missing_20 = pd.read_excel(path + f"{period}-cut-missing.xlsx")
        alarms_20_filled_rows = pd.read_excel(path + f"{period}-cut.xlsx")

        alarms_20_filled_rows = alarms_20_filled_rows.loc[
            alarms_20_filled_rows['Was Missing'] != True]

        missing_20 = missing_20.loc[
            missing_20['Was Missing'] == True]

        alarms_20_filled_rows = alarms_20_filled_rows.append(
            missing_20,
            ignore_index=True,
            sort=False).sort_values(
                ['StationNr', 'TimeOn']).reset_index(drop=True)

    # fill 20 alarms
    else:
        alarms_20 = alarms[(alarms.Alarmcode == 20) | (
            alarms.Alarmcode == 25)].copy()

        alarms_20['Parameter'] = alarms_20[
            'Parameter'].str.replace(' ', '')

        alarms_20 = alarms_20.sort_values(['StationNr', 'TimeOn'])

        alarms_20['Parameter'] = alarms_20['Alarmcode'].map({20: 'Resumed',
                                                             25: 'Stopped'}
                                                            )

        alarms_20_filled_rows = alarms_20.groupby(
            'StationNr').apply(lambda df: fill_20_apply(df,
                                                        period,
                                                        df.name,
                                                        alarms_result_sum))

        alarms_20_filled_rows.reset_index(inplace=True)
        alarms_20_filled_rows = alarms_20_filled_rows.drop(
            'level_1', axis=1)

        missing_20 = alarms_20_filled_rows.groupby('StationNr').apply(
            lambda df: df.loc[((
                df['Was Missing'] == True) | (
                df['Was Missing'].shift() == True) | (
                df['Was Missing'].shift(-1) == True))])

        del missing_20['StationNr']
        missing_20 = missing_20.reset_index(level=0)

        alarms_20_filled_rows.to_excel(path + f"{period}-cut.xlsx")
        missing_20.to_excel(path + f"{period}-cut-missing.xlsx")
    return alarms_20_filled_rows


def upsample_115_20(df_group, period, alarmcode):

    StationId, df = df_group
    df = df.loc[:, ['TimeOn']]

    clmn_name = f'Duration {alarmcode}(s)'
    df[clmn_name] = 0

    df = df.set_index('TimeOn')
    df = df.sort_index()

    df.iloc[::2, 0] = 1
    df.iloc[1::2, 0] = -1

    # precision en miliseconds
    precision = 1000

    df = df.resample(f'{precision}ms', label='right').sum().cumsum()

    df.loc[df[clmn_name] > 0, clmn_name] = precision / 1000

    df = df.resample('10T', label='right').sum()

    period_dt = dt.strptime(period, '%Y-%m')
    period_dt_upper = period_dt + relativedelta(months=1)
    period_dt_upper = period_dt_upper.strftime('%Y-%m')

    full_range = pd.date_range(pd.Timestamp(f'{period}-01 00:10:00.000'),
                               pd.Timestamp(
                                   f'{period_dt_upper}-01 00:00:00.000'),
                               freq='10T')

    df = df.reindex(index=full_range, fill_value=0)

    df['StationId'] = StationId
    return df


class read_files():

    # ------------------------------grd---------------------------------
    @staticmethod
    def read_grd(period):
        usecols_grd = '''TimeStamp, StationId, wtc_ActPower_min,
            wtc_ActPower_max, wtc_ActPower_mean'''

        sql_grd = f"Select {usecols_grd} FROM tblSCTurGrid;"

        grd = zip_to_df(data_type='grd', sql=sql_grd, period=period)
        grd['TimeStamp'] = pd.to_datetime(
            grd['TimeStamp'], format='%m/%d/%y %H:%M:%S')

        grd.iloc[:, 2:] = grd.astype(float, errors='ignore')

        return grd

    # ------------------------------cnt-------------------------------------
    @staticmethod
    def read_cnt(period):
        usecols_cnt = '''TimeStamp, StationId, wtc_kWG1Tot_accum,
        wtc_kWG1TotE_accum'''

        sql_cnt = f"Select {usecols_cnt} FROM tblSCTurCount;"

        cnt = zip_to_df(data_type='cnt', sql=sql_cnt, period=period)

        cnt['TimeStamp'] = pd.to_datetime(
            cnt['TimeStamp'], format='%m/%d/%y %H:%M:%S')

        return cnt

    # -----------------------------sum---------------------------
    @staticmethod
    def read_sum(period):
        usecols_sum = """
        SELECT CDbl(TimeOn) AS TOn, CDbl(TimeOff) AS TOff,
        StationNr, Alarmcode, ID, Parameter
        FROM tblAlarmLog WHERE TimeOff IS NOT NULL
        union
        SELECT CDbl(TimeOn) AS TOn, TimeOff AS TOff,
        StationNr, Alarmcode, ID, Parameter
        FROM tblAlarmLog WHERE TimeOff IS NULL
        """
        alarms = zip_to_df('sum', usecols_sum, period)

        alarms['TOn'] = sqldate_to_datetime(alarms['TOn'])
        alarms['TOff'] = sqldate_to_datetime(alarms['TOff'])

        alarms.rename(columns={'TOn': 'TimeOn',
                               'TOff': 'TimeOff'}, inplace=True)

        alarms = alarms[alarms.StationNr >= 2307405]

        alarms = alarms[
            alarms.StationNr <= 2307535].reset_index(
            drop=True)

        alarms.dropna(subset=['Alarmcode'], inplace=True)

        alarms.reset_index(drop=True, inplace=True)

        alarms.Alarmcode = alarms.Alarmcode.astype(int)

        return alarms

    # ------------------------------tur---------------------------
    @staticmethod
    def read_tur(period):
        usecols_tur = '''TimeStamp, StationId, wtc_AcWindSp_mean, wtc_AcWindSp_stddev,
        wtc_ActualWindDirection_mean, wtc_ActualWindDirection_stddev'''

        sql_tur = f"Select {usecols_tur} FROM tblSCTurbine;"

        tur = zip_to_df('tur', sql_tur, period)

        return tur

    # ------------------------------met---------------------------
    @staticmethod
    def read_met(period):

        usecols_met = '''TimeStamp, StationId ,met_WindSpeedRot_mean,
        met_WinddirectionRot_mean'''

        sql_met = f"Select {usecols_met} FROM tblSCMet;"

        met = zip_to_df('met', sql_met, period)

        met = met.pivot('TimeStamp', 'StationId', [
                        'met_WindSpeedRot_mean', 'met_WinddirectionRot_mean'])

        met.columns = met.columns.to_flat_index()

        met.reset_index(inplace=True)

        return met

    @staticmethod
    def read_all(period):

        return (read_files.read_met(period), read_files.read_tur(period),
                read_files.read_sum(period), read_files.read_cnt(period),
                read_files.read_grd(period))


def full_calculation(period):

    # reading all files with function
    met, tur, alarms, cnt, grd = read_files.read_all(period)

    # --------------------------error list-------------------------
    error_list = pd.read_excel(
        r'Error_Type_List_Las_Update_151209.xlsx')

    error_list.Number = error_list.Number.astype(int)  # ,errors='ignore'

    error_list.drop_duplicates(subset=['Number'], inplace=True)

    error_list.rename(columns={'Number': 'Alarmcode'}, inplace=True)

    # ------------------------------------------------------
    period_dt = dt.strptime(period, "%Y-%m")
    period_month = period_dt.month
    previous_period_dt = period_dt + relativedelta(months=-1)
    previous_period = previous_period_dt.strftime("%Y-%m")
    next_period_dt = period_dt + relativedelta(months=1)
    next_period = next_period_dt.strftime("%Y-%m")

    try:

        previous_alarms = read_files.read_sum(previous_period)
        alarms = alarms.append(previous_alarms)

    except FileNotFoundError:
        print('Previous mounth alarms File not found')

    # ------------------------------------------------------

    ''' label scada alarms with coresponding error type
    and only keep alarm codes in error list'''
    result_sum = pd.merge(alarms, error_list[[
                          'Alarmcode', 'Error Type', 'UK Text']],
                          on='Alarmcode',
                          how='inner', sort=False)

    # Remove warnings
    result_sum = result_sum.loc[result_sum['Error Type'] != 'W']

    # Determine alarms real periods applying cascade method

    print('All Files Loaded Proceeding to calculations')

    print('Cascade')
    alarms_result_sum = apply_cascade(result_sum)
    print('Cascading done')

    # ----------------openning pool for multiprocessing------------------------
    pool = mp.Pool(processes=(mp.cpu_count() - 1))

    # ------------------115 filling binning -----------------------------------

    print('filling 115')
    alarms_115_filled = fill_115(alarms, period, alarms_result_sum)
    print('115 filled')

    print(f'upsampling 115')
    grp_lst_args = iter([(n, period, '115')
                         for n in alarms_115_filled.groupby('StationNr')])

    alarms_115_filled_binned = pool.starmap(upsample_115_20, grp_lst_args)

    alarms_115_filled_binned = pd.concat(alarms_115_filled_binned)

    alarms_115_filled_binned.reset_index(inplace=True)

    alarms_115_filled_binned.rename(columns={'index': 'TimeStamp'},
                                    inplace=True)

    # -------------------20/25 filling binning --------------------------------

    print('filling 20')
    alarms_20_filled = fill_20(alarms, period, alarms_result_sum)
    print('20 filled')
    grp_lst_args = iter([(n, period, '20-25')
                         for n in alarms_20_filled.groupby('StationNr')])

    print(f'upsampling 20-25')
    alarms_20_filled_binned = pool.starmap(upsample_115_20, grp_lst_args)

    alarms_20_filled_binned = pd.concat(alarms_20_filled_binned)

    alarms_20_filled_binned.reset_index(inplace=True)

    alarms_20_filled_binned.rename(columns={'index': 'TimeStamp'},
                                   inplace=True)

    # Remove previous period alarms
    mask = (alarms_result_sum['TimeOn'].dt.month == period_month) | (
        alarms_result_sum['TimeOff'].dt.month == period_month)

    alarms_result_sum = alarms_result_sum.loc[mask]
    
    pool.close()

    # Binning alarms (old method)

    print('Binning')
    # grp_lst_args = iter([(n, period)
    #                      for n in alarms_result_sum.groupby('StationNr')])

    # alarms_binned = pool.starmap(upsample, grp_lst_args)

    # alarms_binned = pd.concat(alarms_binned)

    # alarms_binned.reset_index(inplace=True)
    # alarms_binned.rename(columns={'index': 'TimeStamp'}, inplace=True)

    # Binning alarms and remove overlap with 1005 (new method)

    alarms_df_clean = alarms_result_sum.loc[(
        alarms_result_sum['RealPeriod'].dt.total_seconds() != 0)].copy()

    alarms_df_1005 = alarms_result_sum.loc[
        (alarms_result_sum['Alarmcode'] == 1005)].copy()
    alarms_df_1005['TimeOff'] = alarms_df_1005['NewTimeOn']
    alarms_df_1005['NewTimeOn'] = alarms_df_1005['TimeOn']

    alarms_df_clean_10min = alarms_df_clean.groupby(
        'StationNr').apply(lambda df: alarms_to_10min(df, period, next_period))

    alarms_df_clean_10min = realperiod_10mins(alarms_df_clean_10min)

    alarms_df_1005_10min = alarms_df_1005.groupby(
        'StationNr').apply(lambda df: alarms_to_10min(df, period, next_period))
    alarms_df_1005_10min = realperiod_10mins(alarms_df_1005_10min)

    alarms_df_clean_10min.reset_index(inplace=True)
    alarms_df_1005_10min.reset_index(inplace=True)

    merged_last_df = pd.concat([alarms_df_clean_10min,
                                alarms_df_1005_10min]).sort_values(
                                    ['StationNr', '10minTimeOn']).reset_index()

    merged_last_df = merged_last_df.drop(
        'index', axis=1).rename(columns={'level_1': 'TimeStamp'})

    df = merged_last_df.groupby('StationNr').apply(
        lambda df: remove_1005_overlap(df, df.name, alarms_df_1005_10min))

    mask_siemens = df['Error Type'] == 1
    mask_tarec = df['Error Type'] == 0
    df.loc[mask_siemens,
           'Period Siemens(s)'] = df.loc[mask_siemens, 'RealPeriod']
    df.loc[mask_tarec, 'Period Tarec(s)'] = df.loc[mask_tarec, 'RealPeriod']

    alarms_binned = df.groupby(['StationNr', 'TimeStamp']).agg(
        {'RealPeriod': 'sum',
         'Period Tarec(s)': 'sum',
         'Period Siemens(s)': 'sum',
         'UK Text': '|'.join, }).reset_index()

    alarms_binned = (
        alarms_binned
        .groupby('StationNr')
        .apply(lambda df: full_range(df, period, next_period))
        .reset_index()
        .rename(columns={'level_1': 'TimeStamp',
                         'StationNr': 'StationId',
                         'Period Tarec(s)': 'Period 0(s)',
                         'Period Siemens(s)': 'Period 1(s)'}))

    alarms_binned['Period 0(s)'] = (alarms_binned['Period 0(s)']
                                    .dt.total_seconds())
    alarms_binned['Period 1(s)'] = (alarms_binned['Period 1(s)']
                                    .dt.total_seconds())
    alarms_binned['RealPeriod'] = (alarms_binned['RealPeriod']
                                   .dt.total_seconds())

    print('Alarms Binned')

    # -------merging cnt, grd, tur, met,upsampled (alarms ,115 and 20/25)------
    # merging upsampled alarms with energy production

    print('merging upsampled alarms with energy production')
    cnt_alarms = pd.merge(alarms_binned, cnt,
                          on=['TimeStamp', 'StationId'],
                          how='left').reset_index(drop=True)

    # merging last dataframe with power
    cnt_alarms_minpwr = pd.merge(cnt_alarms, grd,
                                 on=['TimeStamp', 'StationId'], how='left')

    cnt_alarms_minpwr.reset_index(drop=True, inplace=True)

    # merging last dataframe with 115 upsampled
    cnt_115 = pd.merge(cnt_alarms_minpwr, alarms_115_filled_binned,
                       on=['StationId', 'TimeStamp'],
                       how='left')

    # merging last dataframe with 20/25 upsampled
    cnt_115 = pd.merge(cnt_115, alarms_20_filled_binned,
                       on=['StationId', 'TimeStamp'],
                       how='left')

    # merging last dataframe with turbine windspeed data
    cnt_115 = pd.merge(cnt_115,
                       tur[['TimeStamp',
                            'StationId',
                            'wtc_AcWindSp_mean',
                            'wtc_ActualWindDirection_mean']],
                       on=('TimeStamp', 'StationId'),
                       how='left')

    # merging last dataframe with met mast data
    cnt_115 = pd.merge(cnt_115,
                       met,
                       on='TimeStamp',
                       how='left')

    cnt_115 = cnt_115.fillna(0)

    # -------- operational turbines mask --------------------------------------
    mask_n = ((cnt_115['wtc_kWG1TotE_accum'] > 0) & (
        cnt_115['Period 0(s)'] == 0) & (
        cnt_115['Period 1(s)'] == 0) & (
        cnt_115['wtc_ActPower_min'] > 0) & (
        cnt_115['Duration 115(s)'] == 0) & (
        cnt_115['Duration 20-25(s)'] == 0)
    )

    # -------- operational turbines -------------------------------------------
    cnt_115_n = cnt_115.loc[mask_n].copy()

    cnt_115_n['Correction Factor'] = 0
    cnt_115_n['Available Turbines'] = 0

    Epot = cnt_115_n.groupby(
        'TimeStamp').agg({'wtc_kWG1TotE_accum': ep_cf,
                          'Correction Factor': cf_column,
                          'Available Turbines': 'count'})

    Epot = Epot.rename(columns={'wtc_kWG1TotE_accum': 'Epot'})

    del cnt_115_n['Correction Factor']
    del cnt_115_n['Available Turbines']

    cnt_115_n = pd.merge(cnt_115_n, Epot, on='TimeStamp', how='left')

    cnt_115_n['Epot'] = cnt_115_n['wtc_kWG1TotE_accum']

    cnt_115_no = cnt_115.loc[~mask_n].copy()

    cnt_115_no = pd.merge(cnt_115_no, Epot, on='TimeStamp', how='left')

    mask = (cnt_115_no['Epot'] < cnt_115_no['wtc_kWG1TotE_accum'])

    cnt_115_no.loc[mask, 'Epot'] = cnt_115_no.loc[mask, 'wtc_kWG1TotE_accum']

    cnt_115_final = pd.DataFrame()

    cnt_115_final = cnt_115_no.append(cnt_115_n, sort=False)

    cnt_115_final = cnt_115_final.sort_values(
        ['StationId', 'TimeStamp']).reset_index(drop=True)

    cnt_115_final['EL'] = cnt_115_final['Epot'] - \
        cnt_115_final['wtc_kWG1TotE_accum']

    cnt_115_final['EL'] = cnt_115_final['EL'].clip(lower=0)

    cnt_115_final = cnt_115_final.fillna(0)

    cnt_115_final['ELX'] = ((cnt_115_final['Period 0(s)'] / (
        cnt_115_final['Period 0(s)'] + cnt_115_final['Period 1(s)'])) * (
        cnt_115_final['EL'])).fillna(0)

    cnt_115_final['ELNX'] = ((cnt_115_final['Period 1(s)'] / (
        cnt_115_final['Period 0(s)'] + cnt_115_final['Period 1(s)'])) * (
        cnt_115_final['EL'])).fillna(0)

    cnt_115_final.loc[cnt_115_final['Duration 115(s)'] > 0, 'EL 115'] = (
        cnt_115_final.loc[cnt_115_final['Duration 115(s)'] > 0, 'EL'])

    cnt_115_final['EL 115'] = cnt_115_final['EL 115'].fillna(0)

    cnt_115_final.loc[cnt_115_final['Duration 20-25(s)'] > 0, 'EL 20-25'] = (
        cnt_115_final.loc[cnt_115_final['Duration 20-25(s)'] > 0, 'EL'])

    cnt_115_final['EL 20-25'] = cnt_115_final['EL 20-25'].fillna(0)

    cnt_115_final['EL_115_left'] = cnt_115_final['EL 115'] - (
        cnt_115_final['ELX'] + cnt_115_final['ELNX'])

    max_115_ELX_ELNX = pd.concat(
        [(cnt_115_final['ELX'] + cnt_115_final['ELNX']),
         cnt_115_final['EL 115']], axis=1).max(axis=1)

    cnt_115_final['EL_indefini'] = cnt_115_final['EL'] - max_115_ELX_ELNX
    # -------------------------------------------------------------------------

    def lowind(cnt_115_final):

        # if at least 3 of these conditions are true
        mask_1 = ((
            cnt_115_final['wtc_AcWindSp_mean'] < 4.5) & ((
                cnt_115_final[('met_WindSpeedRot_mean', 246)] < 6) | (
                cnt_115_final[('met_WindSpeedRot_mean', 38)] < 6) | (
                cnt_115_final[('met_WindSpeedRot_mean', 39)] < 6))) & (
            cnt_115_final['EL_indefini'] > 0)

        mask_2 = mask_1.shift().bfill()

        cnt_115_final.loc[mask_1,
                          'EL_wind'] = cnt_115_final.loc[
                              mask_1, 'EL_indefini'].fillna(0)

        cnt_115_final.loc[mask_1,
                          'Duration lowind(s)'] = 600

        cnt_115_final.loc[mask_2 & ~mask_1, 'EL_wind_start'] = (
            cnt_115_final.loc[mask_2 & ~mask_1, 'EL_indefini']).fillna(0)

        cnt_115_final.loc[mask_2 & ~mask_1, 'Duration lowind_start(s)'] = 600
        # ---------------------------------------------------------------------

        mask_3 = ((
            cnt_115_final['Duration 115(s)'] > 0).shift() & (
            cnt_115_final['EL_indefini'] > 0) & (
            cnt_115_final['Duration 115(s)'] == 0))

        cnt_115_final.loc[~mask_1 & ~mask_2 & mask_3, 'EL_alarm_start'] = (
            cnt_115_final.loc[
                ~mask_1 & ~mask_2 & mask_3, 'EL_indefini']).fillna(0)

        cnt_115_final.loc[~mask_1 & ~mask_2 & mask_3,
                          'Duration alarm_start(s)'] = 600

        return cnt_115_final

    cnt_115_final = cnt_115_final.groupby('StationId').apply(lowind).fillna(0)

    cnt_115_final['EL_indefini_left'] = cnt_115_final['EL_indefini'] - (
        cnt_115_final['EL_wind'] +
        cnt_115_final['EL_wind_start'] +
        cnt_115_final['EL_alarm_start'])
    # -------------------------------------------------------------------------

    Ep = cnt_115_final['wtc_kWG1TotE_accum'].sum()
    EL = cnt_115_final['EL'].sum()
    ELX = cnt_115_final['ELX'].sum()
    ELNX = cnt_115_final['ELNX'].sum()
    Epot = cnt_115_final['Epot'].sum()
    # EL_indefini = cnt_115_final['EL_indefini'].sum()

    EL_wind = cnt_115_final['EL_wind'].sum()
    EL_wind_start = cnt_115_final['EL_wind_start'].sum()
    EL_alarm_start = cnt_115_final['EL_alarm_start'].sum()

    MAA_result = round(100 * (Ep + ELX) / (Ep + ELX + ELNX), 2)

    MAA_indefini = round(100 * (Ep + ELX) / (Ep + EL), 2)

    MAA_indefni_adjusted = 100 * (
        Ep + ELX) / (
            Ep + EL - (EL_wind + EL_wind_start + EL_alarm_start))

    print(MAA_result, MAA_indefini, MAA_indefni_adjusted)

    return cnt_115_final


# if __name__ == '__main__':
#     full_calculation('2020-03')
