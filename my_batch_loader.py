import numpy as np
import torch
from utils import pad_monolingual_batch
from torch.utils.data import TensorDataset
from itertools import cycle


class BatchLoader:
    def __init__(self, dataset, batch_size):
        self.languages = dataset.languages
        self.load_sentence = dataset.load_sentence
        self.batch_size = batch_size
        # index from 0 to the length of the train or test set
        self.train_ids = {l: np.arange(len(dataset.train[l])) for l in self.languages}
        self.test_ids = np.arange(len(dataset.test))
        self.pad_index = {l: dataset.vocabs[l].get_pad(l) for l in self.languages}

        self.train_data = {l: [self.load_sentence(l, idx) for idx in self.train_ids[l]] for l in self.languages}
        self.test_data = {l: [self.load_sentence(l, idx, test=True) for idx in self.test_ids] for l in self.languages}

        self.padding_dataset()
        # print(self.train_data['src'][0])  # tensors for length of the train data
        # print(self.train_data['src'][1])  # tensors for the whole indexes of the train data

        self.train_length = {l: len(self.train_ids[l]) for l in self.languages}
        self.test_length = {l: len(self.test_data[l]) for l in self.languages}

        self.train_data = \
            {
                l: TensorDataset(torch.tensor(self.train_data[l][0], dtype=torch.long),
                                 torch.tensor(self.train_data[l][1], dtype=torch.long)) for l in self.languages
            }
        self.test_data = \
            {
                l: TensorDataset(torch.tensor(self.test_data[l][0], dtype=torch.long),
                                 torch.tensor(self.test_data[l][1], dtype=torch.long)) for l in self.languages
            }

        self.data_loader = {'train':
                                {l: torch.utils.data.DataLoader(self.train_data[l],
                                                             batch_size=self.batch_size,
                                                             shuffle=True,
                                                             pin_memory=True)
                                 for l in self.languages},
                            'test':
                                {l: torch.utils.data.DataLoader(self.test_data[l],
                                                             batch_size=self.batch_size,
                                                             shuffle=False,
                                                             pin_memory=True)
                                 for l in self.languages}
                            }

    def padding_dataset(self):
        s_length, s_padded_batches = pad_monolingual_batch(self.train_data['src'], self.pad_index['src'])
        t_length, t_padded_batches = pad_monolingual_batch(self.train_data['tgt'], self.pad_index['tgt'])
        self.train_data = \
            {
                'src': [s_length, s_padded_batches],
                'tgt': [t_length, t_padded_batches]
            }

        s_length, s_padded_batches = pad_monolingual_batch(self.test_data['src'], self.pad_index['src'])
        t_length, t_padded_batches = pad_monolingual_batch(self.test_data['tgt'], self.pad_index['tgt'])
        self.test_data = \
            {
                'src': [s_length, s_padded_batches],
                'tgt': [t_length, t_padded_batches]
            }

    def load_batch(self, test=False):
        if test is False:
            temp_src = next(self.load_monolingual_batch(self.data_loader['train']['src']))
            temp_tgt = next(self.load_monolingual_batch(self.data_loader['train']['tgt']))

            length = \
                {
                    'src': temp_src[0],
                    'tgt': temp_tgt[0]
                }
            batch = \
                {
                    'src': temp_src[1].transpose(0, 1),
                    'tgt': temp_tgt[1].transpose(0, 1)
                }
            return length, batch
        else:
            temp_src = next(self.load_monolingual_batch(self.data_loader['test']['src']))
            temp_tgt = next(self.load_monolingual_batch(self.data_loader['test']['tgt']))

            length = \
                {
                    'src': temp_src[0],
                    'tgt': temp_tgt[0]
                }
            batch = \
                {
                    'src': temp_src[1].transpose(0, 1),
                    'tgt': temp_tgt[1].transpose(0, 1)
                }
            return length, batch

    def load_monolingual_batch(self, data_loader):
        for x in data_loader:
            yield x

