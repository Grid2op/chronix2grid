import os
import pandas as pd

def format_for_kpi(repo_in, timestep, prods_charac):
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
    return df