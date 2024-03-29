import numpy as np
import torch
import os.path
from pathlib import Path

from utils import load_embeddings, load_word2nearest, load_train_and_test
from vocabulary import Vocabulary
from tqdm import tqdm
"""
code is based on https://github.com/migonch/unsupervised_mt/blob/master/unsupervised_mt/dataset.py
"""

class Dataset:
    def __init__(self, corp_paths, emb_paths, pairs_paths, max_length=10, test_size=0.1):
        self.languages = ['src', 'tgt']
        self.corp_paths = {l: p for l, p in zip(self.languages, corp_paths)}
        self.emb_paths = {l: p for l, p in zip(self.languages, emb_paths)}
        self.pairs_paths = {l: p for l, p in zip(self.languages, pairs_paths)}
        self.max_length = max_length
        self.test_size = test_size

        # load sentences, embeddings and saved word2nearest
        self.train, self.test = load_train_and_test(*corp_paths, self.max_length, self.test_size, random_state=42)
        self.word2emb = {l: load_embeddings(self.emb_paths[l]) for l in self.languages}

        # create word2nearest for each language pair, i.e., (a, b) and (b, a) if the files do not exist
        pairs_paths_1 = Path(pairs_paths[0])
        pairs_paths_2 = Path(pairs_paths[1])
        if pairs_paths_1.exists() and pairs_paths_2.exists():
            print('Files already exit.')
            pass
        else:
            print('Creating the word2nearest')
            count = 0
            for l1, l2 in zip(self.languages, self.languages[::-1]):
                self.save_word2nearest(pairs_paths[count], l1, l2)
                count += 1

        self.word2nearest = {l: load_word2nearest(self.pairs_paths[l]) for l in self.languages}

        # create vocabularies for source and target languages
        self.vocabs = {l: Vocabulary([l]) for l in self.languages}
        for l1, l2 in zip(self.languages, self.languages[::-1]):
            for sentence in self.train[l1]:
                for word in sentence.strip().split():
                    if word in self.word2emb[l1]:
                        self.vocabs[l1].add_word(word, l1)
                        self.vocabs[l2].add_word(self.word2nearest[l1][word], l2)

        # embedding matrices
        self.emb_matrix = {
            l: np.array([self.word2emb[l][w.split('-', 1)[1]] for w in self.vocabs[l].index2word], dtype=np.float32)
            for l in self.languages
        }

    def load_sentence(self, language, idx, pad=0, test=False):
        sentence = self.test[idx][0 if language == 'src' else 1] if test else self.train[language][idx]
        return self.vocabs[language].get_indices(sentence, language=language, pad=pad)

    def get_sos_index(self, language):
        return self.vocabs[language].get_sos(language)

    def get_eos_index(self, language):
        return self.vocabs[language].get_eos(language)

    def get_pad_index(self, language):
        return self.vocabs[language].get_pad(language)

    def get_unk_index(self, language):
        return self.vocabs[language].get_unk(language)

    def get_nearest(self, word, l1, l2):
        if word in ['<sos>', 'eos', 'pad', 'unk']:
            return word
        else:
            idx = np.argmax(np.array(list(self.word2emb[l2].values())) @ self.word2emb[l1][word])
            return list(self.word2emb[l2].keys())[idx]

    def save_word2nearest(self, path, l1, l2):
        word2nearest = dict()
        for sentence in tqdm(self.train[l1]):
            for word in sentence.strip().split():
                if word in self.word2emb[l1] and word not in word2nearest:
                    word2nearest[word] = self.get_nearest(word, l1, l2)
        word2nearest = np.array(word2nearest)
        np.save(path, word2nearest)

    def translate_sentence_word_by_word(self, sentence, l1, l2):
        sentence = [self.vocabs[l1].index2word[index].split('-', 1)[1] for index in sentence]
        sentence = [self.word2nearest[l1][word] for word in sentence]
        sentence = [self.vocabs[l2].word2index[l2 + '-' + word] for word in sentence]
        return sentence

    def translate_batch_word_by_word(self, batch, l1, l2):
        device = batch.device
        batch = batch.transpose(0, 1).tolist()
        batch = [self.translate_sentence_word_by_word(s, l1, l2) for s in batch]
        return torch.tensor(batch, dtype=torch.long, device=device).transpose(0, 1)

    def visualize_sentence(self, sentence, language):
        return ' '.join([self.vocabs[language].index2word[index].split('-', 1)[1]
                         for index in sentence
                         if index != self.get_eos_index(language) and index != self.get_pad_index(language)])

    def visualize_batch(self, batch, language):
        batch = batch.transpose(0, 1).cpu().tolist()
        return [self.visualize_sentence(sentence, language) for sentence in batch]
