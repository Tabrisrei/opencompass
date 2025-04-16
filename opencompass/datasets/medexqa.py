import csv
import json
import os.path as osp
from os import environ

import pandas as pd
from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset



@LOAD_DATASET.register_module()
class MedExQADataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, **kwargs):
        path = get_data_path(path)
        dataset = DatasetDict()
        for split in ['dev', 'test']:
            raw_data = []
            filename = osp.join(path, split, f'{name}_{split}.tsv')
            with open(filename, encoding='utf-8') as f:
                df = pd.read_csv(f, 
                                 names=['input', 'A', 'B', 'C', 'D', 'ex0', 'ex1', 'answer'],
                                 sep='\t')
                for _, row in df.iterrows():
                    assert len(row) == 8
                    raw_data.append({
                        'input': row[0],
                        'A': row[1],
                        'B': row[2],
                        'C': row[3],
                        'D': row[4],
                        'target': row[7],
                    })
            dataset[split] = Dataset.from_list(raw_data)
        return dataset