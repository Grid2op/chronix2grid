from abc import ABC, abstractmethod
import json
import os

import numpy as np
import pandas as pd

from chronix2grid.generation.dispatch import utils as du


class ConfigManager(ABC):
    def __init__(self, name, root_directory, input_directories, output_directory,
                 required_input_files=None):
        self.name = name
        self.root_directory = root_directory
        self.input_directories = input_directories
        self.output_directory = output_directory
        self.required_input_files = required_input_files if required_input_files is not None else []

    def is_single_input_dir(self):
        if isinstance(self.input_directories, str):
            return True
        if isinstance(self.input_directories, dict):
            return False
        raise RuntimeError("input_directories must be either a string or a dictionnary")

    def validate_input(self):
        if self.is_single_input_dir():
            directory_to_check = os.path.join(self.root_directory, self.input_directories)
            try:
                files_to_check = os.listdir(directory_to_check)
                for config_file in self.required_input_files:
                    if config_file not in files_to_check:
                        return False
            except FileNotFoundError:
                raise FileNotFoundError(self.error_message())
        else:
            for input_name, input_path in self.input_directories.items():
                input_directory_abs_path = os.path.join(self.root_directory, input_path)
                try:
                    files_to_check = os.listdir(input_directory_abs_path)
                    for config_file in self.required_input_files[input_name]:
                        if config_file not in files_to_check:
                            return False
                except FileNotFoundError:
                    raise FileNotFoundError(self.error_message())
        return True

    def validate_output(self):
        return os.path.exists(os.path.join(self.root_directory, self.output_directory))

    def error_message(self):
        output_directory_abs_path = os.path.join(
            self.root_directory, self.output_directory)
        error_message_header = (
            f"\n{self.name} process requires the following configuration "
            f"(directories and files must exist): \n"
            f"  - Root directory is {self.root_directory}\n"
            f"  - Output directory is {output_directory_abs_path}\n"
        )
        if self.is_single_input_dir():
            input_directory_abs_path = os.path.join(
                self.root_directory, self.input_directories)
            formatted_required_input_files = ""
            for filename in sorted(self.required_input_files):
                formatted_required_input_files += "\n      - " + filename

            error_msg_body = (
                    f"  - Input directory is {input_directory_abs_path}\n"
                    f"    - Expected input files are:" + formatted_required_input_files
            )
        else:
            formatted_required_directories = ""
            for input_name, input_path in self.input_directories.items():
                input_directory_abs_path = os.path.join(self.root_directory, input_path)
                formatted_required_directories += (
                        "\n    - " + input_directory_abs_path +
                        "\n      - with expected input files:"
                )
                for required_file in self.required_input_files[input_name]:
                    formatted_required_directories += "\n        - " + required_file

            error_msg_body = (
                    f"  - Input directories are:" + formatted_required_directories
            )
        return error_message_header + error_msg_body

    def validate_configuration(self):
        if not self.validate_output() or not self.validate_input():
            raise FileNotFoundError(self.error_message())
        return True

    @abstractmethod
    def read_configuration(self):
        pass


class LoadsConfigManager(ConfigManager):
    def __init__(self, name, root_directory, input_directories, output_directory,
                 required_input_files=None):
        super(LoadsConfigManager, self).__init__(name, root_directory, input_directories,
                                                 output_directory, required_input_files)

    def read_configuration(self):
        params_file_path = os.path.join(
            self.root_directory,
            self.input_directories['case'], 'params.json')
        with open(params_file_path, 'r') as json1_file:
            json1_str = json1_file.read()
        params = json.loads(json1_str)
        for key, value in params.items():
            try:
                params[key] = float(value)
            except ValueError:
                params[key] = pd.to_datetime(value, format='%Y-%m-%d')

        # Nt_inter = int(params['T'] // params['dt'] + 1)
        try:
            loads_charac = pd.read_csv(
                os.path.join(self.root_directory, self.input_directories['case'],
                             'loads_charac.csv'),
                sep=',')
            names = loads_charac['name']  # to generate error if separator is wrong

        except:
            loads_charac = pd.read_csv(
                os.path.join(self.root_directory, self.input_directories['case'],
                             'loads_charac.csv'),
                sep=';')

        return params, loads_charac

    def read_specific(self):
        load_weekly_pattern = pd.read_csv(
            os.path.join(self.root_directory, self.input_directories['patterns'],
                         'load_weekly_pattern.csv'))
        return load_weekly_pattern


class ResConfigManager(ConfigManager):
    def __init__(self, name, root_directory, input_directories, output_directory,
                 required_input_files=None):
        super(ResConfigManager, self).__init__(name, root_directory, input_directories,
                                               output_directory, required_input_files)

    def read_configuration(self):
        params_file_path = os.path.join(
            self.root_directory,
            self.input_directories['case'], 'params.json')
        with open(params_file_path, 'r') as params_json:
            params = json.load(params_json)
        for key, value in params.items():
            try:
                params[key] = float(value)
            except ValueError:
                params[key] = pd.to_datetime(value, format='%Y-%m-%d')

        # Nt_inter = int(params['T'] // params['dt'] + 1)
        try:
            prods_charac = pd.read_csv(
                os.path.join(self.root_directory, self.input_directories['case'],
                             'prods_charac.csv'),
                sep=',')
            names = prods_charac['name']  # to generate error if separator is wrong

        except:
            prods_charac = pd.read_csv(
                os.path.join(self.root_directory, self.input_directories['case'],
                             'prods_charac.csv'),
                sep=';')

        return params, prods_charac

    def read_specific(self):
        solar_pattern = np.load(
            os.path.join(self.root_directory, self.input_directories['patterns'],
                         'solar_pattern.npy'))
        return solar_pattern



class DispatchConfigManager(ConfigManager):
    def __init__(self, name, root_directory, input_directories, output_directory,
                 required_input_files=None):
        super(DispatchConfigManager, self).__init__(name, root_directory, input_directories,
                                                 output_directory, required_input_files)

    def read_configuration(self):
        self.validate_configuration()
        params_filepath = os.path.join(
            self.root_directory,
            self.input_directories['params'],
            'params_opf.json')
        with open(params_filepath, 'r') as opf_param_json:
            params_opf = json.load(opf_param_json)
        try:
            if params_opf['mode_opf'] == '':
                params_opf['mode_opf'] = None
        except KeyError:
            raise KeyError('The mode_opf field of params_opf.json is missing.')
        bool = False
        try:
            bool = params_opf["loss_grid2op_simulation"]
        except:
            print('Warning: The loss_grid2op_simulation field of params_opf.json is missing. Continuing assuming it is False')
            params_opf["loss_grid2op_simulation"] = False
        if bool:
            oblig_keys = ["idxSlack","nameSlack","pmin_margin","pmax_margin","rampup_margin","rampdown_margin","agent_type"]
            for key in oblig_keys:
                try:
                    params_opf[key]
                except KeyError:
                    raise KeyError('loss_grid2op_simulation is set to True, key '+str(key) + ' must be provided')

        return params_opf


def read_all_configuration(files, root_directory, input_directories):
    params = {}
    for file in files:
        params_file_path = os.path.join(
            root_directory,
            input_directories['case'], file)
        with open(params_file_path, 'r') as params_json:
            params_to_add = json.load(params_json)
        for key, value in params_to_add.items():
            try:
                params_to_add[key] = float(value)
            except ValueError:
                params_to_add[key] = pd.to_datetime(value, format='%Y-%m-%d')
        params.update(params_to_add)
    return params

class LossConfigManager(ConfigManager):
    def __init__(self, name, root_directory, input_directories, output_directory,
                 required_input_files=None):
        super(LossConfigManager, self).__init__(name, root_directory, input_directories,
                                                 output_directory, required_input_files)

    def read_configuration(self):
        self.validate_configuration()
        params_filepath = os.path.join(
            self.root_directory,
            self.input_directories['params'],
            'params_loss.json')
        with open(params_filepath, 'r') as loss_param_json:
            params_loss = json.load(loss_param_json)
        try:
            if params_loss['loss_pattern'] == '':
                params_loss['loss_pattern'] = 'loss_pattern.csv'
        except KeyError:
            raise KeyError('The loss_pattern field of params_loss.json is missing.')
        return params_loss
