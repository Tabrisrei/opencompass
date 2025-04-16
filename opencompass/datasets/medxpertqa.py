import csv
import json
import os.path as osp
from os import environ

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MedXpertQADataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, **kwargs):
        path = get_data_path(path)
        # label_map = {
        #     1: 'A',
        #     2: 'B',
        #     3: 'C',
        #     4: 'D',
        # }

        dataset = DatasetDict()
        # for split in ['dev', 'train', 'test']:
        for split in ['test', 'dev']:
            raw_data = []
            # if split == 'dev':
            #     filename = osp.join(path, 'JMED.jsonl')
            # elif split == 'test':
            filename = osp.join(path, f'{split}.jsonl')
            # print(f'Loading {filename}')
            with open(filename, encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line)
                    try:
                        raw_data.append({
                            'input': line['question'].split('\n')[0],
                            'A': line['options']['A'],
                            'B': line['options']['B'],
                            'C': line['options']['C'],
                            'D': line['options']['D'],
                            'E': line['options']['E'],
                            'F': line['options']['F'],
                            'G': line['options']['G'],
                            'H': line['options']['H'],
                            'I': line['options']['I'],
                            'J': line['options']['J'],
                            'target': line['label'],
                        })
                    except:
                        print(f"Error processing line: {line}")
                print(f"length of loaded raw_data {split}: {len(raw_data)}")
                dataset[split] = Dataset.from_list(raw_data)
        return dataset


