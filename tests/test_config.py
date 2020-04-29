import json
import os
from pathlib import Path
import tempfile
import unittest

from chronix2grid.config import DispatchConfigManager


class TestConfigManager(unittest.TestCase):
    def setUp(self) -> None:
        self.root_directory = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.root_directory, 'input1'))
        os.makedirs(os.path.join(self.root_directory, 'input2'))
        os.makedirs(os.path.join(self.root_directory, 'output'))
        config_file1 = Path(os.path.join(self.root_directory, 'input1', 'config1.json'))
        config_file1.touch()
        config_file2 = Path(os.path.join(self.root_directory, 'input2', 'config2.json'))
        config_file2.touch()
        self.config_manager_multi = DispatchConfigManager(
            name='TestConfig',
            root_directory=self.root_directory,
            input_directories=dict(first_dir='input1', second_dir='input2'),
            output_directory='output',
            required_input_files=dict(first_dir=['config1.json'],
                                      second_dir=['config2.json'])
        )
        os.makedirs(os.path.join(self.root_directory, 'input'))
        config_file = Path(os.path.join(self.root_directory, 'input', 'config.json'))
        config_file.touch()
        self.config_manager_single = DispatchConfigManager(
            name='TestConfig',
            root_directory=self.root_directory,
            input_directories='input',
            output_directory='output',
            required_input_files=['config.json']
        )

    def test_error_message(self):
        input_directory_abs_path = os.path.join(
            self.root_directory, self.config_manager_single.input_directories)
        output_directory_abs_path = os.path.join(
            self.root_directory, self.config_manager_single.output_directory)
        expected_error_message_single = (
            f"\nTestConfig process requires the following configuration "
            f"(directories and files must exist): \n"
            f"  - Root directory is {self.root_directory}\n"
            f"  - Output directory is {output_directory_abs_path}\n"
            f"  - Input directory is {input_directory_abs_path}\n"
            f"    - Expected input files are:\n"
            f"      - config.json"
        )

        expected_error_message_multi = (
            f"\nTestConfig process requires the following configuration "
            f"(directories and files must exist): \n"
            f"  - Root directory is {self.root_directory}\n"
            f"  - Output directory is {output_directory_abs_path}\n"
            f"  - Input directories are:"
            f"\n    - {os.path.join(self.root_directory, 'input1')}"
            f"\n      - with expected input files:"
            f"\n        - config1.json"
            f"\n    - {os.path.join(self.root_directory, 'input2')}"
            f"\n      - with expected input files:"
            f"\n        - config2.json"
        )
        self.assertEqual(
            self.config_manager_single.error_message(),
            expected_error_message_single
        )
        self.assertEqual(
            self.config_manager_multi.error_message(),
            expected_error_message_multi
        )

    def test_input_mode(self):
        self.assertTrue(self.config_manager_single.is_single_input_dir())
        self.assertFalse(self.config_manager_multi.is_single_input_dir())

    def test_raise_input_directory_does_not_exists(self):
        self.config_manager_single.input_directories = 'input_with_typo'
        with self.assertRaises(FileNotFoundError) as cm:
            self.config_manager_single.validate_input()
        self.assertEqual(str(cm.exception), self.config_manager_single.error_message())

    def test_validate_output_directory_fails(self):
        self.config_manager_single.output_directory = 'output_with_typo'
        self.assertFalse(self.config_manager_single.validate_output())

    def test_has_required_inputs(self):
        self.assertTrue(self.config_manager_single.validate_input())
        Path(os.path.join(self.root_directory, 'input', 'randomfile.json')).touch()
        self.assertTrue(self.config_manager_single.validate_input())

        self.config_manager_single.required_input_files.append('missingconfig.json')
        self.assertFalse(self.config_manager_single.validate_input())

    def test_check_configuration(self):
        self.config_manager_single.output_directory = 'output_with_typo'
        with self.assertRaises(FileNotFoundError) as cm:
            self.config_manager_single.validate_configuration()
        self.assertEqual(str(cm.exception), self.config_manager_single.error_message())

        self.config_manager_single.output_directory = 'output'
        self.config_manager_single.input_directories = 'input_with_typo'
        with self.assertRaises(FileNotFoundError) as cm:
            self.config_manager_single.validate_configuration()
        self.assertEqual(str(cm.exception), self.config_manager_single.error_message())

        self.config_manager_single.input_directories = 'input'
        self.config_manager_single.required_input_files = ['config_with_typo.json']
        with self.assertRaises(FileNotFoundError) as cm:
            self.config_manager_single.validate_configuration()
        self.assertEqual(str(cm.exception), self.config_manager_single.error_message())

    def test_read_config(self):
        self.config_manager_multi.input_directories.update(params='input1')
        self.config_manager_multi.required_input_files.update(params=['params_opf.json'])
        params_opf_filepath = os.path.join(self.root_directory, 'input1', 'params_opf.json')
        Path(params_opf_filepath).touch()
        with open(params_opf_filepath, 'w') as params_opf_file:
            json.dump(dict(test=3, mode_opf=''), params_opf_file)
        json_dict = self.config_manager_multi.read_configuration()
        self.assertEqual(json_dict["test"], 3)
        self.assertTrue(json_dict['mode_opf'] is None)






