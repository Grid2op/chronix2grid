import os
import pandas as pd


def eco2mix_to_kpi(kpi_input_folder, timestep, prods_charac, loads_charac, year):
    repo_in = os.path.join(kpi_input_folder, 'eco2mix', 'eCO2mix_RTE_Annuel-Definitif_'+str(year)+'.csv')

    gens = {}
    gens['solar'] = prods_charac[prods_charac['type'] == 'solar']['name'].values
    gens['wind'] = prods_charac[prods_charac['type'] == 'wind']['name'].values
    gens['hydro'] = prods_charac[prods_charac['type'] == 'hydro']['name'].values
    gens['nuclear'] = prods_charac[prods_charac['type'] == 'nuclear']['name'].values
    gens['thermal'] = prods_charac[prods_charac['type'] == 'thermal']['name'].values

    corresp_carrier = {'thermal':['Gaz','Fioul','Bioénergies','Charbon'],
                       'solar':['Solaire'],
                       'wind':['Eolien'],
                       'nuclear': ['Nucléaire'],
                       'hydro':['Hydraulique','Pompage']}

    eco2mix = pd.read_csv(repo_in, sep = ';', encoding = 'latin1', decimal = ',')
    df = eco2mix[['Date', 'Heures', 'Fioul', 'Charbon','Gaz','Bioénergies', 'Nucléaire','Eolien', 'Solaire', 'Hydraulique','Pompage']]

    # Time formatting
    df['Space'] = ' '
    df['Time'] = df['Date']+df['Space']+df['Heures']
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', drop=False, inplace=True)

    # Production formatting
    for carrier_out in corresp_carrier.keys():
        df[carrier_out] = df[corresp_carrier[carrier_out]].sum(axis = 1)

    # Equitable repartition on usecase generators
    for carrier in gens.keys():
        n = len(gens[carrier])
        for col in gens[carrier]:
            df[col] = df[carrier]/n

    df.drop(columns=['Space', 'Date', 'Heures',
                     'Fioul', 'Charbon', 'Gaz', 'Bioénergies', 'Nucléaire', 'Eolien', 'Solaire', 'Hydraulique',
                     'Pompage']+list(corresp_carrier.keys()), inplace=True)

    # Resampling
    df = df.resample(timestep).first()

    # Load computation
    loads = loads_charac['name'].unique()
    agg_conso = df.sum(axis = 1).values

    # Equitable repartition on loads nodes
    conso = pd.DataFrame({'Time': df['Time']})
    for col in loads:
        conso[col] = agg_conso/len(loads)

    return df, conso


def chronics_to_kpi(year, n_scenario, repo_in, timestep, thermal = True):

    print(" Formatting chronics for KPI")

    if thermal:
        price = pd.DataFrame()
        prod_p = pd.DataFrame()
        load_p = pd.DataFrame()

        for month in range(1, 13):
            print('Month number ' + str(month))

            folder = os.path.join(repo_in, str(year), 'Scenario_'+str(n_scenario), str(year)+'_'+str(month))
            prod_p_ = pd.read_csv(os.path.join(folder,'prod_p.csv.bz2'), sep = ';', decimal = '.')
            load_p_ = pd.read_csv(os.path.join(folder,'load_p.csv.bz2'), sep = ';', decimal = '.')
            price_ = pd.read_csv(os.path.join(folder,'price.csv'), sep = ';', decimal = '.')

            for df in load_p_, prod_p_:
                df.rename(columns = {'datetime':'Time'}, inplace = True)

            prod_p = prod_p.append(prod_p_)
            load_p = load_p.append(load_p_)
            price = price.append(price_)
        price['Time'] = load_p['Time']
    else:
        print('Year '+str(year))
        folder = os.path.join(repo_in, 'dispatch', str(year), 'Scenario_' + str(n_scenario))
        solar_p = pd.read_csv(os.path.join(folder, 'solar_p.csv.bz2'), sep=',', decimal='.')

        wind_p = pd.read_csv(os.path.join(folder, 'wind_p.csv.bz2'), sep=',', decimal='.')
        wind_p.drop(columns = ['datetime'],inplace = True)


        load_p = pd.read_csv(os.path.join(folder, 'load_p.csv.bz2'), sep=',', decimal='.')

        prod_p = pd.concat([solar_p, wind_p], axis=1)

        for df in load_p, prod_p:
            df.rename(columns={'datetime': 'Time'}, inplace=True)

    # Optional resampling
    load_p['Time'] = pd.to_datetime(load_p['Time'])
    load_p.set_index('Time', drop=False, inplace=True)
    load_p = load_p.resample(timestep).first()

    prod_p['Time'] = pd.to_datetime(prod_p['Time'])
    prod_p.set_index('Time', drop=False, inplace=True)
    prod_p = prod_p.resample(timestep).first()


    if not thermal:
        return prod_p, load_p
    if thermal:
        return prod_p, load_p, price