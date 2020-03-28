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
    column = column.fillna(0)

    column.loc[~column.isna()] = (
        dt(1899, 12, 30) +
        day_parts[1].astype('timedelta64[D]', errors='ignore') +
        (day_parts[0] * 86400000).astype('timedelta64[ms]', errors='ignore')
    )
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
    print('upsmapling')
    df = df[['NewTimeOn', 'TimeOff', 'Period Siemens(s)', 'Period Tarec(s)']]

    df[['Period Siemens(s)', 'Period Tarec(s)']] = df[[
        'Period Siemens(s)', 'Period Tarec(s)']].fillna(pd.Timedelta(0))

    result = pd.melt(df,
                     id_vars=['Period Tarec(s)', 'Period Siemens(s)'],
                     value_name='date')

    result['Period 0(s)'] = result['variable'].map(
        {'NewTimeOn': 1, 'TimeOff': -1})
    result['Period 1(s)'] = result['Period 0(s)']
    #
    result['Period 0(s)'] *= result[
        'Period Tarec(s)'].dt.total_seconds().fillna(0).astype(int)
    result['Period 1(s)'] *= result[
        'Period Siemens(s)'].dt.total_seconds().fillna(0).astype(int)

    result = result.set_index('date')

    result = result.sort_index()

    # precision en miliseconds
    precision = 10000

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


def CF(M, WTN=131, AL_ALL=0.08):

    def AL(M):
        return AL_ALL * (M - 1) / (WTN - 1)

    return round((1 - AL_ALL) / (1 - AL(M)), 4)


def ep_cf(x):
    M = len(x)
    x = x.mean()
    x = round(x * CF(M), 2)
    return x


def outer_fill(df, period):
    period_dt = dt.strptime(period, '%Y-%m')
    period_dt_upper = period_dt + relativedelta(months=1)
    period_upper = period_dt_upper.strftime('%Y-%m')

    if df['Parameter'].iat[0] == 'Resumed':
        first_row = pd.DataFrame(
            {'TimeOn': pd.Timestamp(f'{period}-01 00:10:00.000'),
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

    # load adjusted 115 alamrs
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

    # fill 115 alarms
    else:
        alarms_115 = alarms[alarms.Alarmcode == 115].copy()

        alarms_115['Parameter'] = alarms_115[
            'Parameter'].str.replace(' ', '')

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

    if df['Parameter'].iat[0] == 'Resumed':
        first_row = pd.DataFrame(
            {'TimeOn': pd.Timestamp(f'{period}-01 00:10:00.000'),
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

    print(f'upsampling {alarmcode}')
    StationId, df = df_group
    df = df.loc[:, ['TimeOn']]

    clmn_name = f'Duration {alarmcode}(s)'
    df[clmn_name] = 0

    df = df.set_index('TimeOn')
    df = df.sort_index()

    df.loc[::2, clmn_name] = 1
    df.loc[1::2, clmn_name] = -1

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


def full_calculation(period):

    # ------------------------------grd---------------------------------
    usecols_grd = '''TimeStamp, StationId, wtc_ActPower_min,
        wtc_ActPower_max, wtc_ActPower_mean'''

    sql_grd = f"Select {usecols_grd} FROM tblSCTurGrid;"

    grd = zip_to_df(data_type='grd', sql=sql_grd, period=period)
    grd['TimeStamp'] = pd.to_datetime(
        grd['TimeStamp'], format='%m/%d/%y %H:%M:%S')

    grd.iloc[:, 2:] = grd.astype(float, errors='ignore')
    # ------------------------------cnt-------------------------------------
    usecols_cnt = 'TimeStamp, StationId, wtc_kWG1Tot_accum, wtc_kWG1TotE_accum'

    sql_cnt = f"Select {usecols_cnt} FROM tblSCTurCount;"

    cnt = zip_to_df(data_type='cnt', sql=sql_cnt, period=period)

    cnt['TimeStamp'] = pd.to_datetime(
        cnt['TimeStamp'], format='%m/%d/%y %H:%M:%S')

    # -----------------------------sum---------------------------
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

    alarms.rename(columns={'TOn': 'TimeOn', 'TOff': 'TimeOff'}, inplace=True)

    alarms = alarms[alarms.StationNr >= 2307405]

    alarms = alarms[
        alarms.StationNr <= 2307535].reset_index(
        drop=True)

    print('Alarms Loaded')

    alarms.dropna(subset=['Alarmcode'], inplace=True)

    alarms.reset_index(drop=True, inplace=True)

    alarms.Alarmcode = alarms.Alarmcode.astype(int)

    # --------------------------error list-------------------------
    error_list = pd.read_excel(
        r'Error_Type_List_Las_Update_151209.xlsx')

    error_list.Number = error_list.Number.astype(int)  # ,errors='ignore'

    error_list.drop_duplicates(subset=['Number'], inplace=True)

    error_list.rename(columns={'Number': 'Alarmcode'}, inplace=True)

    # ------------------------------tur---------------------------
    usecols_tur = '''TimeStamp, StationId, wtc_AcWindSp_mean, wtc_AcWindSp_stddev,
    wtc_ActualWindDirection_mean, wtc_ActualWindDirection_stddev'''

    sql_tur = f"Select {usecols_tur} FROM tblSCTurbine;"

    tur = zip_to_df('tur', sql_tur, period)

    # ------------------------------met---------------------------

    usecols_met = '''TimeStamp, StationId ,met_WindSpeedRot_mean,
     met_WinddirectionRot_mean'''

    sql_met = f"Select {usecols_met} FROM tblSCMet;"

    met = zip_to_df('met', sql_met, period)

    met = met.pivot('TimeStamp', 'StationId', [
                    'met_WindSpeedRot_mean', 'met_WinddirectionRot_mean'])

    met.columns = met.columns.to_flat_index()

    met.reset_index(inplace=True)

    # ------------------------------------------------------

    ''' label scada alarms with coresponding error type
    and only keep alarm codes in error list'''
    result_sum = pd.merge(alarms, error_list[[
                          'Alarmcode', 'Error Type']],
                          on='Alarmcode',
                          how='inner', sort=False)

    # Remove warnings
    result_sum = result_sum.loc[result_sum['Error Type'] != 'W']

    # Determine alarms real periods applying cascade method

    print('All Files Loaded Proceeding to calculations')

    print('Cascade')
    alarms_result_sum = apply_cascade(result_sum)
    print('Cascading done')

    # Binning alarms

    pool = mp.Pool(processes=(mp.cpu_count() - 1))

    print('Binning')
    grp_lst_args = iter([(n, period)
                         for n in alarms_result_sum.groupby('StationNr')])

    alarms_binned = pool.starmap(upsample, grp_lst_args)

    alarms_binned = pd.concat(alarms_binned)

    alarms_binned.reset_index(inplace=True)
    alarms_binned.rename(columns={'index': 'TimeStamp'}, inplace=True)

    print('Alarms Binned')

    # ------------------115 filling binning -----------------------------------

    print('filling 115')
    alarms_115_filled = fill_115(alarms, period, alarms_result_sum)
    print('115 filled')

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

    alarms_20_filled_binned = pool.starmap(upsample_115_20, grp_lst_args)

    alarms_20_filled_binned = pd.concat(alarms_20_filled_binned)

    alarms_20_filled_binned.reset_index(inplace=True)

    alarms_20_filled_binned.rename(columns={'index': 'TimeStamp'},
                                   inplace=True)
    pool.close()
    # -------merging cnt, grd, tur, met,upsampled (alarms ,115 and 20/25)------
    # merging upsampled alarms with energy production

    print('merging upsampled alarms with energy production')
    cnt_alarms = pd.merge(alarms_binned, cnt,
                          on=['TimeStamp', 'StationId'],
                          how='right').reset_index(drop=True)

    # merging last dataframe with power
    cnt_alarms_minpwr = pd.merge(cnt_alarms, grd,
                                 on=['TimeStamp', 'StationId'], how='left')

    cnt_alarms_minpwr.reset_index(drop=True, inplace=True)

    # merging last dataframe with 115 upsampled
    cnt_115 = pd.merge(cnt_alarms_minpwr, alarms_115_filled_binned,
                       on=['StationId', 'TimeStamp'])

    # merging last dataframe with 20/25 upsampled
    cnt_115 = pd.merge(cnt_115, alarms_20_filled_binned,
                       on=['StationId', 'TimeStamp'])

    # merging last dataframe with turbine windspeed data
    cnt_115 = pd.merge(cnt_115,
                       tur[['TimeStamp',
                            'StationId',
                            'wtc_AcWindSp_mean',
                            'wtc_ActualWindDirection_mean']],
                       on=('TimeStamp', 'StationId'))

    # merging last dataframe with met mast data
    cnt_115 = pd.merge(cnt_115,
                       met,
                       on='TimeStamp')

    # -------- operational turbines mask --------------------------------------
    mask_n = ((cnt_115['wtc_kWG1Tot_accum'] > 0) & (
        cnt_115['Period 0(s)'] == 0) & (
        cnt_115['Period 1(s)'] == 0) & (
        cnt_115['wtc_ActPower_min'] > 0) & (
        cnt_115['Duration 115(s)'] == 0) & (
        cnt_115['Duration 20-25(s)'] == 0)
    )

    # -------- operational turbines -------------------------------------------
    cnt_115_n = cnt_115.loc[mask_n].copy()

    Epot = cnt_115_n.groupby(
        ['TimeStamp']).agg({'wtc_kWG1Tot_accum': ep_cf}).rename(
            columns={'wtc_kWG1Tot_accum': 'Epot'})

    cnt_115_n['Epot'] = cnt_115_n['wtc_kWG1Tot_accum']

    cnt_115_no = cnt_115.loc[~mask_n].copy()

    cnt_115_no = pd.merge(cnt_115_no, Epot, on='TimeStamp', how='left')

    cnt_115_final = pd.DataFrame()

    cnt_115_final = cnt_115_no.append(cnt_115_n, sort=False)

    cnt_115_final = cnt_115_final.sort_values(
        ['StationId', 'TimeStamp']).reset_index(drop=True)

    cnt_115_final['EL'] = cnt_115_final['Epot'] - \
        cnt_115_final['wtc_kWG1Tot_accum']

    cnt_115_final['EL'] = cnt_115_final['EL'].clip(lower=0)

    cnt_115_final = cnt_115_final.fillna(0)

    cnt_115_final['ELX'] = ((cnt_115_final['Period 0(s)']) / 600) * (
        cnt_115_final['EL'])

    cnt_115_final['ELNX'] = ((cnt_115_final['Period 1(s)']) / 600) * (
        cnt_115_final['EL'])

    cnt_115_final['EL 115'] = ((cnt_115_final['Duration 115(s)']) / 600) * (
        cnt_115_final['EL'])

    cnt_115_final['EL_indefini'] = cnt_115_final['EL'] - (
        cnt_115_final['ELX'] + cnt_115_final['ELNX'])

    Ep = cnt_115_final['wtc_kWG1Tot_accum'].sum()
    ELX = cnt_115_final['ELX'].sum()
    ELNX = cnt_115_final['ELNX'].sum()
    Epot = cnt_115_final['Epot'].sum()
    EL115 = cnt_115_final['EL 115'].sum()
    EL_indefini = cnt_115_final['EL_indefini'].sum()

    MAA_result = round(100 * (Ep + ELX) / (Ep + ELX + ELNX), 2)

    MAA_115 = round(100 * (Ep + ELX) / (Ep + EL115), 2)

    print(MAA_result, MAA_115)
    return cnt_115_final
