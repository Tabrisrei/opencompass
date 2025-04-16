import csv
import json
import os.path as osp
from os import environ

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MedQADataset(BaseDataset):

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
        # for split in ['dev', 'train', 'test']:
        for split in ['test', 'dev']:
            raw_data = []
            filename = osp.join(path, f'{split}.json')
            print(f'Loading {filename}')
            with open(filename, encoding='utf-8') as f:
                # lines = json.load(f)
                # fix the error:json.decoder.JSONDecodeError: Extra data: line 2 column 1 (char 574)
                for line in f:
                    line = json.loads(line)
                    raw_data.append({
                        'input': line['sent1'],
                        'A': line['ending0'],
                        'B': line['ending1'],
                        'C': line['ending2'],
                        'D': line['ending3'],
                        'target': label_map[line['label']],
                    })
            dataset[split] = Dataset.from_list(raw_data)
        return dataset


