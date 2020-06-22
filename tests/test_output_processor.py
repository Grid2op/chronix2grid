import os
import tempfile
import unittest

import pandas as pd
import pathlib

from chronix2grid.output_processor import (dataframe_cutter,
                                           cut_csv_file_into_chunks,
                                           save_chunks)


class TestOutputProcessor(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(index=range(4*8), columns=['a', 'b'],
                          data=[[i, 8*i] for i in range(4*8)])

        self.file_path = tempfile.mkstemp(suffix='.csv.bz2')[1]
        self.df.to_csv(self.file_path, sep=',', index=False)

    def test_dataframe_cutter(self):
        cut_dataframe = dataframe_cutter(self.df, chunk_size=4)
        self.assertEqual(len(cut_dataframe), 8)
        self.assertEqual(len(cut_dataframe[4]), 4)
        self.assertEqual(cut_dataframe[7].iloc[0, 0], 28)

    def test_dataframe_cutter(self):
        df = pd.concat([self.df, pd.DataFrame(index=[-1], columns=self.df.columns,
                                             data=[[-1, -2]])])
        cut_dataframe = dataframe_cutter(df, chunk_size=4)
        self.assertEqual(len(cut_dataframe), 9)
        self.assertEqual(len(cut_dataframe[0]), 4)
        self.assertEqual(len(cut_dataframe[-1]), 1)

    def test_cut_csv_file_into_chunks(self):
        cut_dataframe = cut_csv_file_into_chunks(self.file_path, chunk_size=4, sep=',')
        self.assertEqual(len(cut_dataframe), 8)
        self.assertEqual(len(cut_dataframe[4]), 4)
        self.assertEqual(cut_dataframe[7].iloc[0, 0], 28)

    def test_save_chunks(self):
        save_chunks(
            cut_csv_file_into_chunks(self.file_path, chunk_size=4, sep=','),
            self.file_path
        )
        original_file_name = pathlib.Path(self.file_path).name
        parent_dir = pathlib.Path(self.file_path).parent.absolute()
        df = pd.read_csv(
            os.path.join(parent_dir, 'chunk_0', original_file_name), sep=',')
        self.assertEqual(len(df), 4)
        self.assertEqual(df.iloc[0, 0], 0)
