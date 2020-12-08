import os
import pandas as pd

def main(input_folder, output_folder, load, prod_solar, prod_wind, params, params_loss, write_results = True):
    """
    :param input_folder (str): input folder in which pattern folder can be found
    :param output_folder (str): output folder to write loss chronic
    :param load(pandas.DataFrame): generated load with Chronix2grid module (for other usage)
    :param prod_solar (pandas.DataFrame): generated solar production with Chronix2grid module (for other usage)
    :param prod_wind (pandas.DataFrame): generated wind production with Chronix2grid module (for other usage)
    :param params (dict): dictionary with parameters concerning chronic generations in general
    :param params_loss (dict): dictionary with parameters concerning loss computation
    :param write_results (bool): whether to serialize results or not in scenario_folder_path (True by default)
    :return: pandas.Series representing provided  loss chronic
    """

    loss_pattern_path = os.path.join(input_folder, 'patterns', params_loss["loss_pattern"])
    loss = generate_valid_loss(loss_pattern_path, params)
    if write_results:
        loss.to_csv(os.path.join(output_folder,'loss.csv.bz2'), sep = ';')
    return loss

def generate_valid_loss(loss_pattern_path, params):
    loss = pd.read_csv(loss_pattern_path, sep = ';')
    # TODO: Check longueur, changer date ? S'inspirer largement. Supposer que les jour-mois sont contenus dedans
    return loss

def check_chronix(scenario_folder_path):
    bool = True
    if os.path.exists(scenario_folder_path):
        if not (os.path.exists(os.path.join(scenario_folder_path,"load_p.csv.bz2") and
                               os.path.exists(os.path.join(scenario_folder_path,"prod_p.csv.bz2")))):
            bool = False
    else:
         bool = False
    return bool

