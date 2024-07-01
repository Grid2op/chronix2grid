import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import fastparquet as fp
import logging
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_folder",
    type=str,
    help="The directory in which the parquet files are saved",
)

parser.add_argument(
    "--filename",
    type=str,
    help="The name of the final parquet file",
    default= None
)

def merge_parquet_files(data_folder, filename, logger=None):
    list_files = os.listdir(data_folder)
    if len(list_files)==0:
        raise Exception("No files found in data_folder %s", data_folder)
    logger.debug("%s files found in data_folder '%s'", len(list_files), data_folder)
    dirs_files=sorted(set([el for el in list_files if ".parq" in el]))
    
    logger.info("%s parquet files are going to be merged", len(dirs_files))
    
    parquet_file = os.path.join(data_folder,filename)
    try:
        if os.path.exists(parquet_file):
            raise Exception("Merged file already exists")
        total_rows = 0
        for file in tqdm(dirs_files):
            logger.info("loading %s", file)
            append = os.path.exists(parquet_file)
            
            df = fp.ParquetFile(os.path.join(data_folder, file)).to_pandas()
            total_rows += df.shape[0]
            logger.info("merging %s", file)
            fp.write(parquet_file, df, append=append, compression = 'ZSTD', stats=False)
            logger.info("success in merging %s, target file now contains %s individuals", file, total_rows)
        
        if os.path.exists(parquet_file) and total_rows != 0:    
            logger.info('Full parquet file named "%s" has been written to disk with %s total rows', filename, total_rows)            
            
    except Exception as e:
        logging.error(str(e))

if __name__ == "__main__":
    args = parser.parse_args()
    
    logname = f"Outputs/parquetfiles_merging_{datetime.now()}.log"
    if args.filename:
        filename = args.filename+"_allscenarios.parq"
    else:
        filename = args.data_folder.split("Outputs_")[-1]+"_allscenarios.parq"
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename = logname,
                    level = logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    logging.info("Looking for file to merge in '%s'", args.data_folder)
    dirs_files=sorted(set([el for el in os.listdir(args.data_folder) if os.path.isdir(os.path.join(args.data_folder, el)) and ".parq" in el]))
    
    merge_parquet_files(data_folder=args.data_folder, filename=filename, logger=logging)