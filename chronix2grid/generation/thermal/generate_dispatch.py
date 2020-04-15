import os
import shutil

import numpy as np

from . import dispatch_utils as disp
from .EDispatch_L2RPN2020 import run_economic_dispatch


def main(dispatcher, input_folder, output_folder, seed, params, lines,
         params_opf, compute_hazards=True):

    np.random.seed(seed)

    hydro_constraints = dispatcher.make_hydro_constraints_from_res_load_scenario()
    agg_load_without_renew = dispatcher.net_load(params_opf['losses_pct'],
                                                 name=dispatcher.loads.index[0])

    dispatch_results = dispatcher.run(
        agg_load_without_renew,
        params=params_opf,
        gen_constraints=hydro_constraints,
        ramp_mode=run_economic_dispatch.RampMode.none,
        by_carrier=params_opf['dispatch_by_carrier']
    )

    dispatcher.save_results(output_folder)

    print('Generating maintenance and hazard signals ...')
    if compute_hazards:
        maintenance = disp.compute_random_event('maintenance', lines, params)
        hazards = disp.compute_random_event('hazards', lines, params)
        disp.create_csv(maintenance, os.path.join(output_folder, 'maintenance.csv.bz2'), reordering=True)
        disp.create_csv(maintenance, os.path.join(output_folder, 'maintenance_forecasted.csv.bz2'), reordering=True)
        disp.create_csv(hazards, os.path.join(output_folder, 'hazards.csv.bz2'), reordering=True)

    shutil.copy(os.path.join(input_folder, 'load_p_forecasted.csv.bz2'),
                os.path.join(output_folder, 'load_p_forecasted.csv.bz2'))
    shutil.copy(os.path.join(input_folder, 'wind_p_forecasted.csv.bz2'),
                os.path.join(output_folder, 'wind_p_forecasted.csv.bz2'))
    shutil.copy(
        os.path.join(input_folder, 'solar_p_forecasted.csv.bz2'),
        os.path.join(output_folder, 'solar_p_forecasted.csv.bz2'))

    return dispatch_results
