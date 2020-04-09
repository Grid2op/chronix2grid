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
    try:
        prods_charac = pd.read_csv(os.path.abspath(generation_input_folder+'/'+case+'/prods_charac.csv'), sep=',')
        names = prods_charac['name']
        loads_charac = pd.read_csv(os.path.abspath(generation_input_folder+'/'+case+'/loads_charac.csv'), sep=',')
    except:
        prods_charac = pd.read_csv(os.path.abspath(generation_input_folder + '/' + case + '/prods_charac.csv'), sep=';')
        loads_charac = pd.read_csv(os.path.abspath(generation_input_folder + '/' + case + '/loads_charac.csv'), sep=';')

    for scenario_num in range(n_scenarios):
        print('Scenario '+str(scenario_num)+'...')
        if params['weeks'] != 52:
            print('Warning: KPI are incomplete. Computation has been made on '+str(params['weeks'])+' weeks, but are meant to be computed on 52 weeks')
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
            ref_dispatch, ref_consumption, syn_dispatch, syn_consumption, monthly_pattern, hours, ref_prices, prices = pivot_format(
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


        syn_dispatch.index = ref_dispatch.index
        syn_consumption.index = ref_dispatch.index
        ref_consumption.index = ref_dispatch.index

        # Load agg price profile
        if not wind_solar_only:

            # Start Economic dispatch validator
            # -- + -- + -- + -- + -- + -- + --
            print ('(1) Computing KPI\'s...')
            dispatch_validator = EconomicDispatchValidator(ref_consumption,
                                                           syn_consumption,
                                                           ref_dispatch,
                                                           syn_dispatch,
                                                           year,
                                                           scenario_num,
                                                           images_repo,
                                                           prods_charac=prods_charac,
                                                           loads_charac=loads_charac,
                                                           ref_prices=ref_prices,
                                                           syn_prices=prices)
            dispatch_validator.energy_mix()

        else:
            # Start Economic dispatch validator
            # -- + -- + -- + -- + -- + -- + --
            print('(1) Computing KPI\'s...')
            dispatch_validator = EconomicDispatchValidator(ref_consumption,
                                                           syn_consumption,
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
        dispatch_validator.plot_carriers_pw(curve='reference', stacked=True, max_col_splot=max_col, save_html=True,
                                            wind_solar_only=wind_solar_only)
        dispatch_validator.plot_carriers_pw(curve='synthetic', stacked=True, max_col_splot=max_col, save_html=True,
                                            wind_solar_only=wind_solar_only)

        # Get Load KPI
        dispatch_validator.load_kpi()

        # These KPI only if dispatch has been made
        if not wind_solar_only:
            # Get Hydro KPI
            dispatch_validator.hydro_kpi()

            # Get Nuclear KPI
            dispatch_validator.nuclear_kpi()

            # Get Thermal KPI
            dispatch_validator.thermal_kpi()
            dispatch_validator.thermal_load_kpi()


        # Get Wind KPI
        dispatch_validator.wind_kpi()

        # Get Solar KPI
        dispatch_validator.solar_kpi(monthly_pattern = monthly_pattern, hours = hours)

        # Wind - Solar KPI
        dispatch_validator.wind_load_kpi()


        # Write json output file
        # -- + -- + -- + -- + --
        print ('(2) Generating json output file...')

        kpi_output_folder = os.path.join('kpi/output',str(year))
        if not os.path.exists(kpi_output_folder):
            os.mkdir(kpi_output_folder)
        kpi_output_folder = os.path.join(kpi_output_folder,'Scenario_' + str(scenario_num))
        if not os.path.exists(kpi_output_folder):
            os.mkdir(kpi_output_folder)

        with open(os.path.join(kpi_output_folder,'ec_validator_output.json'), 'w') as json_f:
            json.dump(dispatch_validator.output, json_f)

        print ('-Done-\n')
        

if __name__=="__main__":
    main()