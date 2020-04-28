import datetime as dt
import math
import os

import pandas as pd
import pathlib

from .generation import generation_utils as gu


def write_start_dates_for_chunks(output_path, n_weeks, by_n_weeks,
                                 n_scenarios, start_date):
    days_to_day = by_n_weeks * 7
    n_chunks = math.ceil(n_weeks / by_n_weeks)
    start_date_time = pd.to_datetime(start_date, format='%Y-%m-%d')
    file_name = 'start_datetime.info'
    scen_name_generator = gu.folder_name_pattern('Scenario', n_scenarios)
    chunk_folder_name_generator = gu.folder_name_pattern('chunk', n_chunks)
    for j in range(n_scenarios):
        scenario_name = scen_name_generator(j)
        write_start_datetime_info(
            os.path.join(output_path, scenario_name),
            file_name,
            start_date_time)
        for i in range(n_chunks):
            chunk_folder_name = chunk_folder_name_generator(i)
            output_directory = os.path.join(
                output_path, scenario_name, chunk_folder_name)
            write_start_datetime_info(output_directory, file_name, start_date_time)
            start_date_time += dt.timedelta(days=days_to_day)


def write_start_datetime_info(directory_path, file_name, start_date_time):
    with open(os.path.join(directory_path, file_name), 'w') as f:
        f.write(start_date_time.strftime('%Y-%m-%d %H:%M'))


def output_processor_to_chunks(output_path, by_n_weeks, n_scenarios):
    chunk_size = by_n_weeks * 7 * 24 * 12  # 5 min time step
    scen_name_generator = gu.folder_name_pattern('Scenario', n_scenarios)
    for i in range(n_scenarios):
        scenario_name = scen_name_generator(i)
        csv_files_to_process = os.listdir(os.path.join(output_path, scenario_name))
        csv_files_to_process = [
            os.path.join(output_path, scenario_name, csv_file) for csv_file in csv_files_to_process
            if os.path.isfile(os.path.join(output_path, scenario_name, csv_file))
        ]
        generate_chunks(csv_files_to_process, chunk_size)


def generate_chunks(csv_files_to_process, chunk_size, sep=','):
    for csv_file in csv_files_to_process:
        cut_df = cut_csv_file_into_chunks(csv_file, chunk_size, sep=sep)
        save_chunks(cut_df, csv_file, index=False)


def cut_csv_file_into_chunks(csv_file_path, chunk_size, **kwargs):
    df = pd.read_csv(csv_file_path, **kwargs)
    return dataframe_cutter(df, chunk_size)


def save_chunks(chunks, original_file_path, **kwargs):
    parent_dir = pathlib.Path(original_file_path).parent.absolute()
    original_file_name = pathlib.Path(original_file_path).name
    chunk_folder_name_generator = gu.folder_name_pattern('chunk', len(chunks))
    for i, chunk in enumerate(chunks):
        chunk_folder_name = chunk_folder_name_generator(i)
        os.makedirs(os.path.join(parent_dir, chunk_folder_name), exist_ok=True)
        chunk.to_csv(
            os.path.join(parent_dir, chunk_folder_name, original_file_name),
            **kwargs
        )


def dataframe_cutter(df, chunk_size):
    """
    Cut a csv file into chunks
    Parameters
    ----------
    df pandas.DataFrame
        The DataFrame to cut
    chunk_size int
        The size of each chunk

    Returns
    -------
        The list of cut dataframes
    """
    cut_df = []
    n_chunks = len(df) // chunk_size
    for i in range(n_chunks):
        cut_df.append(df.iloc[i*chunk_size:((i+1)*chunk_size)])
    if len(df) > chunk_size * n_chunks:
        cut_df.append(df.iloc[chunk_size*n_chunks:])
    return cut_df
