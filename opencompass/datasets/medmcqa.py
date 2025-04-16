import csv
import json
import os.path as osp
from os import environ

from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MedMCQADataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, **kwargs):
        path = get_data_path(path)
        label_map = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D',
        }
        dataset = DatasetDict()
        for split in ['train', 'validation']:

            raw_data = []
            filename = osp.join(path, f'{split}-00000-of-00001.parquet')
        
            dataset_origin = load_dataset('parquet', data_files=filename, split='train')
            # reformat the dataset
            for data in dataset_origin:
                raw_data.append({
                    'input': data['question'],
                    'A': data['opa'],
                    'B': data['opb'],
                    'C': data['opc'],
                    'D': data['opd'],
                    'target': label_map[data['cop']],
                })
            # from pprint import pprint
            # pprint(f"raw_data 0: {raw_data[0]}")
            # print(f"length of raw_data: {len(raw_data)}")
            dataset[split] = Dataset.from_list(raw_data)
        return dataset