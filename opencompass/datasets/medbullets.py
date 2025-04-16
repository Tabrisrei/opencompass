import csv
import json
import os.path as osp
from os import environ

from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MedbulletsOp5Dataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, **kwargs):
        path = get_data_path(path)

        dataset = DatasetDict()
        raw_data = []
        
        filename = osp.join(path, 'train-00000-of-00001.parquet')
        dataset = load_dataset('parquet', data_files=filename, split='train')
        # reformat the dataset
        for data in dataset:
            raw_data.append({
                'input': data['question'],
                'A': data['opa'],
                'B': data['opb'],
                'C': data['opc'],
                'D': data['opd'],
                'E': data['ope'],
                'target': data['answer_idx'],
            })
        from pprint import pprint
        # pprint(f"raw_data 0: {raw_data[0]}")
        
        # print(f"length of raw_data: {len(raw_data)}")
        dataset = Dataset.from_list(raw_data)
        return dataset
