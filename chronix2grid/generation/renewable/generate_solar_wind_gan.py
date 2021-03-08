import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Other Python libraries
import pandas as pd
import numpy as np

# Libraries developed for this module
from . import solar_wind_utils as swutils
import chronix2grid.constants as cst
from chronix2grid.generation.renewable.gan_utils import ReplayedGAN, generate_gaussian_inputs, post_process_sample, load_wind_model

import tensorflow as tf


def main_gan(scenario_destination_path, seed, params, prods_charac, write_results = True):


    np.random.seed(seed)

    network_folder = params["network_folder"]

    # Define datetime indices
    datetime_index = pd.date_range(
        start=params['start_date'],
        end=params['end_date'],
        freq=str(params['dt']) + 'min')

    # Read GAN network session
    sess = tf.InteractiveSession()
    dcgan_model = load_wind_model(sess, params, network_folder)

    # Generate random inputs
    Y,Z = generate_gaussian_inputs(params)

    # Simulate with GAN
    Z_tf_sample, Y_tf_sample, image_tf_sample = dcgan_model.samples_generator(batch_size=params["batch_size"])
    generated_batches = []
    for y, z in zip(Y,Z):
        generated_samples = sess.run(
            image_tf_sample,
            feed_dict={
                Z_tf_sample: z,
                Y_tf_sample: y
            })
        generated_batches.append(generated_samples)
    wind_series = post_process_sample(generated_batches, params, prods_charac, datetime_index, carrier="wind")

    ################### A changer
    solar_gens = prods_charac[prods_charac['type']=='solar']['name'].unique()
    solar_series_matrix = np.ones((len(datetime_index),len(solar_gens)))
    solar_series = pd.DataFrame(solar_series_matrix, columns=solar_gens)
    prods_series = pd.concat([wind_series,solar_series], axis = 1)
    solar_series['datetime'] = datetime_index
    ##########################

    # Save files
    print('Saving files in zipped csv')
    if not os.path.exists(scenario_destination_path):
        os.makedirs(scenario_destination_path)

    prod_solar_forecasted =  swutils.create_csv(
        solar_series,
        os.path.join(scenario_destination_path, 'solar_p_forecasted.csv.bz2'),
        reordering=True,
        shift=True,
        write_results=write_results,
        index=False
    )

    prod_solar = swutils.create_csv(
        solar_series,
        os.path.join(scenario_destination_path, 'solar_p.csv.bz2'),
        reordering=True,
        noise=params['planned_std'],
        write_results=write_results
    )

    prod_wind_forecasted = swutils.create_csv(
        wind_series,
        os.path.join(scenario_destination_path, 'wind_p_forecasted.csv.bz2'),
        reordering=True,
        shift=True,
        write_results=write_results,
        index=False
    )

    prod_wind = swutils.create_csv(
        wind_series, os.path.join(scenario_destination_path, 'wind_p.csv.bz2'),
        reordering=True,
        noise=params['planned_std'],
        write_results=write_results
    )

    prod_p = swutils.create_csv(
        prods_series, os.path.join(scenario_destination_path, 'prod_p.csv.bz2'),
        reordering=True,
        noise=params['planned_std'],
        write_results=write_results
    )

    prod_v = prods_charac[['name', 'V']].set_index('name')
    prod_v = prod_v.T
    prod_v.index = [0]
    prod_v = prod_v.reindex(range(len(prod_p)))
    prod_v = prod_v.fillna(method='ffill') * 1.04

    prod_v.to_csv(
        os.path.join(scenario_destination_path, 'prod_v.csv.bz2'),
        sep=';',
        index=False,
        float_format=cst.FLOATING_POINT_PRECISION_FORMAT
    )

    return prod_solar, prod_solar_forecasted, prod_wind, prod_wind_forecasted


