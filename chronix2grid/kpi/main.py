import argparse
import warnings
import os
import json
import pandas as pd

# import sys
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

from kpi.preprocessing.pivot_KPI import pivot_format
from kpi.deterministic.kpis import EconomicDispatchValidator



def main(kpi_input_folder, generation_input_folder, generation_output_folder, images_repo, year, case, n_scenarios, wind_solar_only, params):


    print('=====================================================================================================================================')
    print('================================================= KPI GENERATION  ===================================================================')
    print('=====================================================================================================================================')

    warnings.filterwarnings("ignore")

    # Get grid charac
    prods_charac = pd.read_csv(os.path.abspath(generation_input_folder+'/'+case+'/prods_charac.csv'), sep=';')
    loads_charac = pd.read_csv(os.path.abspath(generation_input_folder+'/'+case+'/loads_charac.csv'), sep=';')

    for scenario_num in range(n_scenarios):
        print('Scenario '+str(scenario_num)+'...')
        if wind_solar_only:
            # Get reference and synthetic dispatch and loads
            ref_dispatch, ref_consumption, syn_dispatch, syn_consumption, monthly_pattern, hours = pivot_format(generation_input_folder,
                                                                                                            kpi_input_folder,
                                                                                                            year,
                                                                                                            scenario_num,
                                                                                                            prods_charac,
                                                                                                            loads_charac,
                                                                                                            wind_solar_only,
                                                                                                                params)
        else:
            # Get reference and synthetic dispatch and loads
            ref_dispatch, ref_consumption, syn_dispatch, syn_consumption, monthly_pattern, hours, prices = pivot_format(
                                                                                                generation_output_folder,
                                                                                                kpi_input_folder,
                                                                                                year,
                                                                                                scenario_num,
                                                                                                prods_charac,
                                                                                                loads_charac,
                                                                                                wind_solar_only,
                                                                                                params)


        # -- + -- + ---
        # DONT FORGET TO ADD THE CONSTRAINT THE REF DISPATCH IT'S ONLY
        # UNITL XXXX-12-31 11:00 TO THE OTHERS DATAFRAMES / ANO BICIESTO
        # -- + -- + ---

        # Set same index as ref dispatch to avoid
        # problem when working in posterior KPI's

        # =================== Temporaire A SUPPRIMER ==================================== !!!!!!!!!!!!!!!
        ref_dispatch = ref_dispatch.head(len(syn_dispatch))
        syn_consumption = syn_consumption.head(len(syn_dispatch))
        # =============================================================================== !!!!!!!!!!!!!!!

        syn_dispatch.index = ref_dispatch.index
        syn_consumption.index = ref_dispatch.index

        # Load agg price profile
        if not wind_solar_only:
            # prices = pd.read_csv(os.path.abspath(kpi_input_folder+'/'+str(year)+'+/Scenario_'+str(scenario_num)+'/prices.csv'), sep = ';')
            prices['Time'] = pd.to_datetime(prices['Time'])
            prices.set_index('Time', drop=True, inplace=True)

            # Start Economic dispatch validator
            # -- + -- + -- + -- + -- + -- + --
            print ('(1) Computing KPI\'s...')
            dispatch_validator = EconomicDispatchValidator(syn_consumption,
                                                           ref_dispatch,
                                                           syn_dispatch,
                                                           year,
                                                           scenario_num,
                                                           images_repo,
                                                           prods_charac=prods_charac,
                                                           loads_charac=loads_charac,
                                                           prices=prices)
            dispatch_validator.energy_mix()

        else:
            # Start Economic dispatch validator
            # -- + -- + -- + -- + -- + -- + --
            print('(1) Computing KPI\'s...')
            dispatch_validator = EconomicDispatchValidator(syn_consumption,
                                                           ref_dispatch,
                                                           syn_dispatch,
                                                           year,
                                                           scenario_num,
                                                           images_repo,
                                                           prods_charac=prods_charac,
                                                           loads_charac=loads_charac)

        # Get dispatch temporal view
        if wind_solar_only:
            max_col = 1
        else:
            max_col = 2
        # dispatch_validator.plot_carriers_pw(curve='reference', stacked=True, max_col_splot=max_col, save_html=True,
        #                                     wind_solar_only=wind_solar_only)
        # dispatch_validator.plot_carriers_pw(curve='synthetic', stacked=True, max_col_splot=max_col, save_html=True,
        #                                     wind_solar_only=wind_solar_only)

        # Get Hydro KPI
        if not wind_solar_only:
            dispatch_validator.hydro_kpi()

        # Get Wind KPI
        dispatch_validator.wind_kpi()

        # Get Solar KPI
        dispatch_validator.solar_kpi(monthly_pattern = monthly_pattern, hours = hours)
        # dispatch_validator.solar_kpi()

        # Wind - Solar KPI
        dispatch_validator.wind_load_kpi()

        # Get Nuclear KPI
        if not wind_solar_only:
            dispatch_validator.nuclear_kpi()

        # Write json output file
        # -- + -- + -- + -- + --
        # print ('(2) Generating json output file...')
        # with open(os.path.join(args.destination, 'ec_validator_output.txt'), 'w') as json_f:
        #     json.dump(dispatch_validator.output, json_f)

        print ('-Done-\n')
        

if __name__=="__main__":
    main()