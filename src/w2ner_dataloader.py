import logging
import pickle
import re
from itertools import repeat
from os.path import join
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from ee_data import NER_PAD, EEDataloader, InputExample

logger = logging.getLogger(__name__)

LABEL = ['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'sym', 'dis', 'bod']
W2_ID2LABEL = ['[PAD]', 'suc'] + LABEL
W2_LABEL2ID = {L: i for i, L in enumerate(W2_ID2LABEL)}
W2_NUM_LABELS = len(W2_ID2LABEL)

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class W2NERDataset(Dataset):
    def __init__(self, cblue_root: str, mode: str, max_length: int, tokenizer, for_nested_ner: bool):
        self.cblue_root = cblue_root
        self.data_root = join(cblue_root, "CMeEE")
        self.max_length = max_length
        self.for_nested_ner = for_nested_ner
        self.mode = mode

        # This flag is used in CRF
        self.no_decode = mode.lower() == "train"

        # Processed data can vary from using different tokenizers
        _tk_class = re.match("<class '.*\.(.*)'>", str(type(tokenizer))).group(1)
        _head = 2 if for_nested_ner else 1
        cache_file = join(self.data_root, f"cache_{mode}_{max_length}_{_tk_class}_{_head}head.pkl")

        if False and exists(cache_file):
            with open(cache_file, "rb") as f:
                self.examples, self.data = pickle.load(f)
            logger.info(f"Load cached data from {cache_file}")
        else:
            self.examples = EEDataloader(cblue_root).get_data(mode)  # get original data
            self.data = self._preprocess(self.examples, tokenizer)  # preprocess
            with open(cache_file, 'wb') as f:
                pickle.dump((self.examples, self.data), f)
            logger.info(f"Cache data to {cache_file}")

    def _preprocess(self, examples: List[InputExample], tokenizer) -> list:
        """NOTE: This function is what you need to modify for W2NER.
        """
        is_test = examples[0].entities is None
        data = []

        for example in examples:
            if is_test:
                _sentence_id, text = example.to_begin_end_label_tuples()
                # labels = None
                label = repeat(None, len(text))
            else:
                _sentence_id, text, label = example.to_begin_end_label_tuples()

            text = text[:self.max_length - 2]

            length = len(text)
            _grid_labels = np.zeros((length, length), dtype=np.int)
            _dist_inputs = np.zeros((length, length), dtype=np.int)
            _grid_mask2d = np.ones((length, length), dtype=np.bool)

            tokens = []
            for word in text:
                token = tokenizer.tokenize(word)
                if not token:
                    token = [tokenizer.unk_token]
                # assert len(token) == 1
                tokens.extend(token)

            tokens = [tokenizer.cls_token] + tokens[: self.max_length - 2] + [tokenizer.sep_token]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            for k in range(length):
                _dist_inputs[k, :] += k
                _dist_inputs[:, k] -= k

            for i in range(length):
                for j in range(length):
                    if _dist_inputs[i, j] < 0:
                        _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                    else:
                        _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
            _dist_inputs[_dist_inputs == 0] = 19

            if not is_test:
                for start, end, lid in label:
                    if start <= end < self.max_length - 2:
                        for idx in range(start, end):
                            _grid_labels[idx, idx + 1] = W2_LABEL2ID['suc']
                        _grid_labels[end, start] = lid
            data.append((token_ids, length, _dist_inputs, _grid_mask2d, _grid_labels))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.no_decode


class fn_N2WER:
    def __init__(self, pad_token_id: int, label_pad_token_id: int = W2_LABEL2ID[NER_PAD]):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, batch) -> dict:
        input = [b[0] for b in batch]
        no_decode = batch[0][1]

        bert_inputs, length, dist_inputs, grid_mask2d, grid_labels = map(list, zip(*input))
        grid_labels = grid_labels if grid_labels[0] is not None else None

        max_len = max(length)
        max_input_len = max(len(ids) for ids in bert_inputs)

        attention_mask = torch.zeros((len(batch), max_input_len), dtype=torch.long)
        for id, token in enumerate(bert_inputs):
            attention_mask[id][:len(token)] = 1
            _delta_len = max_input_len - len(token)
            bert_inputs[id] += [self.pad_token_id] * _delta_len

        dist_inputs = self.padding(dist_inputs, 0, max_len)
        grid_mask2d = self.padding(grid_mask2d, False, max_len)

        if grid_labels is not None:
            grid_labels = self.padding(grid_labels, self.label_pad_token_id, max_len)

        return {
            "input_ids": torch.tensor(bert_inputs, dtype=torch.long),
            "attention_mask": attention_mask.long(),
            "length": torch.tensor(length, dtype=torch.long),
            "dist_inputs": torch.tensor(dist_inputs, dtype=torch.long),
            "grid_mask2d": torch.tensor(grid_mask2d, dtype=torch.bool),
            "labels": torch.tensor(grid_labels, dtype=torch.long) if grid_labels is not None else None,
            "no_decode": no_decode
        }

    def padding(self, data, pad_val, target_len) -> np.ndarray:

        n_dims = len(data[0].shape)

        for idx, d in enumerate(data):
            delta_lens = []
            for j in range(n_dims):
                delta_lens.append((0, target_len - d.shape[j]))
            data[idx] = np.pad(d, delta_lens, 'constant', constant_values=pad_val)
        return data
