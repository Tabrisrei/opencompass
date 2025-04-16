import csv
import json
import os.path as osp
from os import environ

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class CareQADataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, **kwargs):
        path = get_data_path(path)
        label_map = {
            1: 'A',
            2: 'B',
            3: 'C',
            4: 'D',
        }

        dataset = DatasetDict()
        # for split in ['dev', 'train', 'test']:
        for split in ['test', 'dev']:
            raw_data = []
            if split == 'dev':
                filename = osp.join(path, 'CareQA_es.json')
            elif split == 'test':
                filename = osp.join(path, 'CareQA_en.json')
            # print(f'Loading {filename}')
            with open(filename, encoding='utf-8') as f:
                lines = json.load(f)
                for line in lines:
                    # print(f"line: {line}")
                # for line in f:
                #     line = json.loads(line)
                    raw_data.append({
                        'input': line['question'],
                        'A': line['op1'],
                        'B': line['op2'],
                        'C': line['op3'],
                        'D': line['op4'],
                        'target': label_map[line['cop']],
                    })
            dataset[split] = Dataset.from_list(raw_data)
        return dataset


