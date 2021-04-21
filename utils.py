import io
import unicodedata
import re
import numpy as np
from typing import List
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

"""
code is based on https://github.com/migonch/unsupervised_mt/blob/master/unsupervised_mt/utils.py
"""

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r'([.!?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    return s.strip()


def load_embeddings(emb_path, encoding='utf-8', newline='\n', errors='ignore'):
    vec_path = emb_path + '.npy'
    path = Path(vec_path)
    if path.exists():
        word2emb = np.load(vec_path, allow_pickle=True)
        return word2emb.item()
    else:
        word2emb = dict()
        with io.open(emb_path, 'r', encoding=encoding, newline=newline, errors=errors) as f:
            emb_dim = int(f.readline().split()[1])
            for word in ['<sos>', '<eos>', '<unk>', '<pad>']:
                word2emb[word] = np.random.uniform(0, 1, size=emb_dim)
                word2emb[word] /= np.linalg.norm(word2emb[word])

            for line in f.readlines()[1:]:
                orig_word, emb = line.rstrip().split(' ', 1)
                emb = np.fromstring(emb, sep=' ')
                word = normalize_string(orig_word)

                # if word is not in dictionary or if it is, but better embedding is provided
                if word not in word2emb or word == orig_word:
                    word2emb[word] = emb
        word2emb = np.array(word2emb)
        np.save(vec_path, word2emb)
        return word2emb.item()


def load_train_and_test(src_path, tgt_path, max_length, test_size, random_state=42,
                        encoding='utf-8', newline='\n', errors='ignore'):
    with io.open(src_path, 'r', encoding=encoding, newline=newline, errors=errors) as f:
        src_sentences = list(map(normalize_string, f.readlines()))
    with io.open(tgt_path, 'r', encoding=encoding, newline=newline, errors=errors) as f:
        tgt_sentences = list(map(normalize_string, f.readlines()))

    assert len(src_sentences) == len(tgt_sentences)

    np.random.seed(random_state)
    train_src, test_src, train_tgt, test_tgt = train_test_split(src_sentences, tgt_sentences, test_size=test_size)
    # print(len(train_src))
    # print(len(test_src))
    test = list(filter(
        lambda p: len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length,
        zip(test_src, test_tgt)
    ))
    train = {l: list(filter(lambda s: len(s.split(' ')) < max_length, sentences))
             for l, sentences in zip(['src', 'tgt'], [train_src, train_tgt])}

    for sentences in train.values():
        np.random.shuffle(sentences)

    return train, test


def load_word2nearest(path):
    word2nearest = dict()
    for word in ['<sos>', '<eos>', '<unk>', '<pad>']:
        word2nearest[word] = word
    word2nearest.update(np.load(path, allow_pickle=True).item())
    return word2nearest


def pad_monolingual_batch(batch: List[int], pad_index):
    max_length = np.max([len(s) for s in batch])
    length = [len(s) for s in batch]  # eos does not count
    return length, [s + (max_length - len(s)) * [pad_index] for s in batch]


def noise_sentence(sentence: List[int], pad_index, drop_probability=0.1, permutation_constraint=3):
    sentence = list(filter(lambda index: index != pad_index, sentence))
    eos = sentence[-1:]
    sentence = sentence[:-1]
    np.random.seed()

    sentence = list(filter(lambda index: np.random.binomial(1, 1 - drop_probability), sentence))

    # notation from paper
    alpha = permutation_constraint + 1
    q = np.arange(len(sentence)) + np.random.uniform(0, alpha, size=len(sentence))
    sentence = list(np.array(sentence)[np.argsort(q)])

    return sentence + eos


def noise(batch: torch.Tensor, pad_index, drop_probability=0.1, permutation_constraint=3):
    device = batch.device
    batch = batch.transpose(0, 1).tolist()
    batch = [noise_sentence(s, pad_index, drop_probability, permutation_constraint) for s in batch]
    length, b = pad_monolingual_batch(batch, pad_index)
    return torch.tensor(length, dtype=torch.long, device=device), \
           torch.tensor(b, dtype=torch.long, device=device).transpose(0, 1)


def log_probs2indices(decoder_outputs):
    return decoder_outputs.topk(k=1)[1].squeeze(-1)
