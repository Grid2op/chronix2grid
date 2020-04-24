# Native python packages
import os

import click

# Chronix2grid modules
from chronix2grid.generation import generate_chronics as gen
from chronix2grid.generation import generation_utils as gu
from chronix2grid.kpi import main as kpis
from chronix2grid.output_processor import (
    output_processor_to_chunks, write_start_dates_for_chunks)

# ==============================================================
## CONSTANT VARIABLES
@click.command()
@click.option('--case', default='case118_l2rpn', help='case folder to base generation on')
@click.option('--start-date', default='2012-01-01', help='Start date to generate chronics')
@click.option('--weeks', default=4, help='Number of weeks to generate')
@click.option('--by-n-weeks', default=4, help='Size of the output chunks in weeks')
@click.option('--n_scenarios', default=1, help='Number of scenarios to generate')
@click.option('--mode', default='LRTK', help='Steps to execute : L for loads only (and KPI); R(K) for renewables (and KPI) only; LRTK for all generation')
@click.option('--root-folder', default=os.path.normpath(os.getcwd()), help='root of all file generation and input')
@click.option('--seed-for-loads', default=None, help='Input seed to ensure reproducibility of loads generation')
@click.option('--seed-for-res', default=None, help='Input seed to ensure reproducibility of renewables generation')
@click.option('--seed-for-dispatch', default=None, help='Input seed to ensure reproducibility of dispatch')
def generate(case, start_date, weeks, by_n_weeks, n_scenarios, mode, root_folder,
             seed_for_loads, seed_for_res, seed_for_dispatch):
    generate_inner(case, start_date, weeks, by_n_weeks, n_scenarios, mode, root_folder,
             seed_for_loads, seed_for_res, seed_for_dispatch)


def generate_inner(case, start_date, weeks, by_n_weeks, n_scenarios, mode, root_folder,
             seed_for_loads, seed_for_res, seed_for_dispatch):
    INPUT_FOLDER = os.path.join(root_folder, 'generation', 'input')
    OUTPUT_FOLDER = os.path.join(root_folder, 'generation', 'output')

    KPI_INPUT_FOLDER = os.path.join(root_folder, "kpi", "input")
    IMAGES_FOLDER = os.path.join(root_folder, "kpi", "images")
    # Folders are specific to studied case
    output_folder = os.path.join(OUTPUT_FOLDER, case)
    images_folder = os.path.join(IMAGES_FOLDER, case)

    time_parameters = gu.time_parameters(weeks, start_date)
    seed_for_loads = parse_seed_arg(seed_for_loads, '--seed-for-loads')
    seed_for_res = parse_seed_arg(seed_for_res, '--seed-for-res')
    seed_for_dispatch = parse_seed_arg(seed_for_dispatch, '--seed-for-dispatch')

    year = time_parameters['year']

    # Chronic generation
    if 'L' in mode or 'R' in mode:
        params, loads_charac, prods_charac = gen.main(
            case, n_scenarios, INPUT_FOLDER, output_folder, time_parameters,
            mode, seed_for_loads, seed_for_res, seed_for_dispatch)
        if by_n_weeks is not None:
            output_processor_to_chunks(OUTPUT_FOLDER, case, year, by_n_weeks, n_scenarios)
            write_start_dates_for_chunks(OUTPUT_FOLDER, case, year, weeks,
                                         by_n_weeks, n_scenarios, start_date)

    # KPI formatting and computing
    if 'R' in mode and 'K' in mode and 'T' not in mode:
        # Get and format solar and wind on all timescale, then compute KPI and save plots
        wind_solar_only = True
        os.makedirs(IMAGES_FOLDER, exist_ok=True)
        kpis.main(KPI_INPUT_FOLDER, INPUT_FOLDER, output_folder, images_folder,
                  year, case, n_scenarios, wind_solar_only, params,
                  loads_charac, prods_charac)

    elif 'T' in mode and 'K' in mode:
        # Get and format dispatched chronics, then compute KPI and save plots
        wind_solar_only = False
        os.makedirs(IMAGES_FOLDER, exist_ok=True)
        kpis.main(KPI_INPUT_FOLDER, INPUT_FOLDER, output_folder, images_folder,
                  year, case, n_scenarios, wind_solar_only, params,
                  loads_charac, prods_charac)


def parse_seed_arg(seed, arg_name):
    if seed is not None:
        try:
            seed = int(seed)
        except TypeError:
            raise RuntimeError(f'The parameter {arg_name} must be an integer')
    return seed


if __name__ == "__main__":
    generate()
