import pandas as pd
import os


def kpi_format_production(year, n_scenario, repo_in, repo_out, timestep, thermal = True):

    print(" Formatting data for KPI")

    if thermal:
        price = pd.DataFrame()
        prod_p = pd.DataFrame()
        load_p = pd.DataFrame()
        prod_p_f = pd.DataFrame()
        load_p_f = pd.DataFrame()
        for month in range(1, 13):
            print('Mois ' + str(month))

            folder = os.path.join(repo_in, str(year), 'Scenario_'+str(n_scenario), str(year)+'_'+str(month))
            prod_p_ = pd.read_csv(os.path.join(folder,'prod_p.csv.bz2'), sep = ';', decimal = '.')
            load_p_ = pd.read_csv(os.path.join(folder,'load_p.csv.bz2'), sep = ';', decimal = '.')
            prod_p_f_ = pd.read_csv(os.path.join(folder,'prod_p_forecasted.csv.bz2'), sep = ';', decimal = '.')
            load_p_f_ = pd.read_csv(os.path.join(folder,'load_p_forecasted.csv.bz2'), sep = ';', decimal = '.')
            price_ = pd.read_csv(os.path.join(folder,'price.csv'), sep = ';', decimal = '.')

            for df in load_p_, prod_p_, prod_p_f_, load_p_f_:
                df.rename(columns = {'datetime':'Time'}, inplace = True)

            prod_p = prod_p.append(prod_p_)
            load_p = load_p.append(load_p_)
            prod_p_f = prod_p_f.append(prod_p_f_)
            load_p_f = load_p_f.append(load_p_f_)
            price = price.append(price_)
        price['Time'] = load_p['Time']
    else:
        print('Année '+str(year))
        folder = os.path.join(repo_in, 'dispatch', str(year), 'Scenario_' + str(n_scenario))
        solar_p = pd.read_csv(os.path.join(folder, 'solar_p.csv.bz2'), sep=',', decimal='.')
        solar_p_f = pd.read_csv(os.path.join(folder, 'solar_p_forecasted.csv.bz2'), sep=',', decimal='.')

        wind_p = pd.read_csv(os.path.join(folder, 'wind_p.csv.bz2'), sep=',', decimal='.')
        wind_p.drop(columns = ['datetime'],inplace = True)
        wind_p_f = pd.read_csv(os.path.join(folder, 'wind_p_forecasted.csv.bz2'), sep=',', decimal='.')
        wind_p_f.drop(columns=['datetime'], inplace=True)

        load_p = pd.read_csv(os.path.join(folder, 'load_p.csv.bz2'), sep=',', decimal='.')
        load_p_f = pd.read_csv(os.path.join(folder, 'load_p_forecasted.csv.bz2'), sep=',', decimal='.')

        prod_p = pd.concat([solar_p, wind_p], axis=1)
        prod_p_f = pd.concat([solar_p_f, wind_p_f], axis=1)

        for df in load_p, prod_p, prod_p_f, load_p_f:
            df.rename(columns={'datetime': 'Time'}, inplace=True)

    # Optional resampling
    load_p['Time'] = pd.to_datetime(load_p['Time'])
    load_p.set_index('Time', drop=False, inplace=True)
    load_p = load_p.resample(timestep).first()
    load_p_f['Time'] = pd.to_datetime(load_p_f['Time'])
    load_p_f.set_index('Time', drop=False, inplace=True)
    load_p_f = load_p_f.resample(timestep).first()

    prod_p['Time'] = pd.to_datetime(prod_p['Time'])
    prod_p.set_index('Time', drop=False, inplace=True)
    prod_p = prod_p.resample(timestep).first()
    prod_p_f['Time'] = pd.to_datetime(prod_p_f['Time'])
    prod_p_f.set_index('Time', drop=False, inplace=True)
    prod_p_f = prod_p_f.resample(timestep).first()

    folder_out = os.path.join(repo_out, str(year))
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)
    folder_out = os.path.join(repo_out, str(year), 'Scenario_'+str(n_scenario))
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)
    prod_p.to_csv(os.path.join(folder_out,'prod_p.csv'), sep = ';', index = False)
    load_p.to_csv(os.path.join(folder_out,'load_p.csv'), sep = ';', index = False)
    prod_p_f.to_csv(os.path.join(folder_out, 'prod_p_forecasted.csv'), sep = ';', index = False)
    load_p_f.to_csv(os.path.join(folder_out, 'load_p_forecasted.csv'), sep = ';', index = False)
    if thermal:
        price.to_csv(os.path.join(folder_out, 'prices.csv'), sep = ';', index = False)


