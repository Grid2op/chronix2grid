import os

import pandas as pd

from chronix2grid import constants
from chronix2grid import utils
from chronix2grid.config import LoadsConfigManager, ResConfigManager, DispatchConfigManager
from chronix2grid.generation import generation_utils
from chronix2grid.generation.consumption.ConsumptionGeneratorBackend import ConsumptionGeneratorBackend
from chronix2grid.generation.dispatch import generate_dispatch, EconomicDispatch
from chronix2grid.generation.dispatch.DispatchBackend import DispatchBackend
from chronix2grid.generation.renewable import generate_solar_wind
from chronix2grid.generation.renewable.RenewableBackend import RenewableBackend


class GeneratorBackend:
    def __init__(self, consumption_backend_class=ConsumptionGeneratorBackend
                 , dispatch_backend_class=DispatchBackend
                 , hydro_backend_class=None
                 , renewable_backend_class=RenewableBackend):
        self.consumption_backend_class = consumption_backend_class
        self.dispatch_backend_class = dispatch_backend_class
        self.hydro_backend_class = hydro_backend_class
        self.renewable_backend_class = renewable_backend_class

    # Call generation scripts n_scenario times with dedicated random seeds
    def run(self, case, n_scenarios, input_folder, output_folder, scen_names,
            time_params, mode='LRTK', scenario_id=None,
            seed_for_loads=None, seed_for_res=None, seed_for_disp=None):
        """
        Main function for chronics generation. It works with three steps: load generation, renewable generation (solar and wind) and then dispatch computation to get the whole energy mix

        Parameters
        ----------
        case (str): name of case to study (must be a folder within input_folder)
        n_scenarios (int): number of desired scenarios to generate for the same timescale
        params (dict): parameters of generation, as returned by function chronix2grid.generation.generate_chronics.read_configuration
        input_folder (str): path of folder containing inputs
        output_folder (str): path where outputs will be written (intermediate folder case/year/scenario will be used)
        prods_charac (pandas.DataFrame): as returned by function chronix2grid.generation.generate_chronics.read_configuration
        loads_charac (pandas.DataFrame): as returned by function chronix2grid.generation.generate_chronics.read_configuration
        lines (pandas.DataFrame): as returned by function chronix2grid.generation.generate_chronics.read_configuration
        solar_pattern (pandas.DataFrame): as returned by function chronix2grid.generation.generate_chronics.read_configuration
        load_weekly_pattern (pandas.DataFrame): as returned by function chronix2grid.generation.generate_chronics.read_configuration
        mode (str): options to launch certain parts of the generation process : L load R renewable T thermal
        generator_backend (GeneratorBackend): The backend class to do all generation parts

        Returns
        -------

        """

        utils.check_scenario(n_scenarios, scenario_id)

        print('=====================================================================================================================================')
        print('============================================== CHRONICS GENERATION ==================================================================')
        print('=====================================================================================================================================')

        # in multiprocessing, n_scenarios=1 here
        if n_scenarios >= 2:
            seeds_for_loads, seeds_for_res, seeds_for_disp = generation_utils.generate_seeds(
                n_scenarios, seed_for_loads, seed_for_res, seed_for_disp
            )
        else:
            seeds_for_loads = [seed_for_loads]
            seeds_for_res = [seed_for_res]
            seeds_for_disp = [seed_for_disp]

        # dispatch_input_folder, dispatch_input_folder_case, dispatch_output_folder = gu.make_generation_input_output_directories(input_folder, case, year, output_folder)
        load_config_manager = LoadsConfigManager(
            name="Loads Generation",
            root_directory=input_folder,
            input_directories=dict(case=case, patterns='patterns'),
            required_input_files=dict(case=['loads_charac.csv', 'params.json'],
                                      patterns=['load_weekly_pattern.csv']),
            output_directory=output_folder
        )
        load_config_manager.validate_configuration()

        params, loads_charac, load_weekly_pattern = load_config_manager.read_configuration()

        res_config_manager = ResConfigManager(
            name="Renewables Generation",
            root_directory=input_folder,
            input_directories=dict(case=case, patterns='patterns'),
            required_input_files=dict(case=['prods_charac.csv', 'params.json'],
                                      patterns=['solar_pattern.npy']),
            output_directory=output_folder
        )

        params, prods_charac, solar_pattern = res_config_manager.read_configuration()

        params.update(time_params)
        params = generation_utils.updated_time_parameters_with_timestep(params, params['dt'])

        dispath_config_manager = DispatchConfigManager(
            name="Dispatch",
            root_directory=input_folder,
            output_directory=output_folder,
            input_directories=dict(params=case),
            required_input_files=dict(params=['params_opf.json'])
        )
        dispath_config_manager.validate_configuration()
        params_opf = dispath_config_manager.read_configuration()
        grid_path = os.path.join(input_folder, case, constants.GRID_FILENAME)
        dispatcher = EconomicDispatch.init_dispatcher_from_config(grid_path, input_folder)

        ## Launch proper scenarios generation
        seeds_iterator = zip(seeds_for_loads, seeds_for_res, seeds_for_disp)

        for i, (seed_load, seed_res, seed_disp) in enumerate(seeds_iterator):

            if n_scenarios > 1:
                scenario_name = scen_names(i)
            else:
                scenario_name = scen_names(scenario_id)

            scenario_folder_path = os.path.join(output_folder, scenario_name)

            print("================ Generating " + scenario_name + " ================")
            if 'L' in mode:
                generator_loads = self.consumption_backend_class(scenario_folder_path, seed_load, params, loads_charac, load_weekly_pattern,
                                                                 write_results=True)
                load, load_forecasted = generator_loads.run()

            if 'R' in mode:
                generator_enr = self.renewable_backend_class(scenario_folder_path, seed_res, params,
                                                             prods_charac,
                                                             solar_pattern, write_results=True)

                prod_solar, prod_solar_forecasted, prod_wind, prod_wind_forecasted = generator_enr.run()
            if 'T' in mode:
                prods = pd.concat([prod_solar, prod_wind], axis=1)
                res_names = dict(wind=prod_wind.columns, solar=prod_solar.columns)
                dispatcher.chronix_scenario = EconomicDispatch.ChroniXScenario(load, prods, res_names,
                                                                               scenario_name)

                dispatch_results = generate_dispatch.main(dispatcher, scenario_folder_path,
                                                          scenario_folder_path,
                                                          seed_disp, params, params_opf)
            print('\n')
        return params, loads_charac, prods_charac
