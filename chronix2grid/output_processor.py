import os

import pandas as pd
import pathlib


def output_processor_to_chunks(intermediary_path, case, year, by_n_weeks, n_scenarios):
    output_path = os.path.join(intermediary_path, case, str(year))
    chunk_size = by_n_weeks * 7 * 24 * 12  # 5 min time step
    for i in range(n_scenarios):
        scenario_name = f"Scenario_{i}"
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
    for i, chunk in enumerate(chunks):
        os.makedirs(os.path.join(parent_dir, f'chunk_{i}'), exist_ok=True)
        chunk.to_csv(
            os.path.join(parent_dir, f'chunk_{i}', original_file_name),
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

