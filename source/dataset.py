#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 19:17:04 2021

@author: nanzheng
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import json

from constants import EOS, PAD


class DatasetWav2Vec2Phn(Dataset):
    """
    
    """
    
    def __init__(self, file_index, file_vocab, path_sample, split="train"):
        self.df_index = pd.read_csv(file_index, sep="\t")
        self.df_index = self.df_index[self.df_index["split"] == split].reset_index(drop=True)
        self.df_index["length"] = self.df_index["target_phoneme"].apply(lambda x: len(x))
        self.df_index = self.df_index.sort_values(by="length", ascending=True).reset_index(drop=True)
        
        with open(file_vocab, "r") as f_vocab:
            self.dict_vocab = json.load(f_vocab)
        
        self.path_sample = os.path.join(path_sample, split)
    
    def __getitem__(self, index):
        file_sample = os.path.join(self.path_sample, "{}.npy".format(str(self.df_index["sample_index"].values[index]).zfill(4)))
        array_feature = np.load(file_sample).astype(np.float32)
        
        target_text = self.df_index["target_phoneme"].values[index][2: -2].split("', '")
        label = np.array(list(map(lambda x: self.dict_vocab[x], target_text)))
        
        return {"feature": array_feature, "label": label}
    
    def __len__(self):
        return len(self.df_index)
    
    def collate(self, batch):
        tensor_feature = pad_sequence(list(map(lambda x: torch.from_numpy(x["feature"]), batch)), batch_first=True)
        tensor_label = pad_sequence(list(map(lambda x: torch.from_numpy(x["label"]), batch)), batch_first=True, padding_value=self.dict_vocab[PAD])
        
        return {"feature": tensor_feature, "label": tensor_label}


class DatasetSeq2SeqPhn(Dataset):
    """
    
    """
    
    def __init__(self, file_index, file_vocab, path_sample, split="train"):
        self.df_index = pd.read_csv(file_index, sep="\t")
        self.df_index = self.df_index[self.df_index["split"] == split].reset_index(drop=True)
        self.df_index["length"] = self.df_index["target_phoneme"].apply(lambda x: len(x))
        self.df_index = self.df_index.sort_values(by="length", ascending=True).reset_index(drop=True)
        # self.df_index = self.df_index[: int(len(self.df_index) * 0.01)]
        
        with open(file_vocab, "r") as f_vocab:
            self.dict_vocab = json.load(f_vocab)
        # del self.dict_vocab[PAD]
        self.dict_vocab[EOS] = len(self.dict_vocab)
        
        self.path_sample = os.path.join(path_sample, split)
    
    def __getitem__(self, index):
        file_sample = os.path.join(self.path_sample, "{}.npy".format(str(self.df_index["sample_index"].values[index]).zfill(4)))
        array_feature = np.load(file_sample).astype(np.float32)
        
        target_text = self.df_index["target_phoneme"].values[index][2: -2].split("', '")
        target_text = target_text[: -1] + [EOS]
        label = np.array(list(map(lambda x: self.dict_vocab[x], target_text)))
        
        return {"feature": array_feature, "label": label}
    
    def __len__(self):
        return len(self.df_index)
    
    def collate(self, batch):
        tensor_feature = pad_sequence(list(map(lambda x: torch.from_numpy(x["feature"]), batch)), batch_first=True)
        tensor_label = pad_sequence(list(map(lambda x: torch.from_numpy(x["label"]), batch)), batch_first=True, padding_value=self.dict_vocab[PAD])
        # tensor_label = pad_sequence(list(map(lambda x: torch.from_numpy(x["label"]), batch)), batch_first=True, padding_value=self.dict_vocab[EOS])
        
        return {"feature": tensor_feature, "label": tensor_label}


class DatasetRawWav2Vec2Phn(Dataset):
    """
    
    """
    
    def __init__(self, file_index, file_vocab, path_sample, split="train"):
        self.df_index = pd.read_csv(file_index, sep="\t")
        self.df_index = self.df_index[self.df_index["split"] == split].reset_index(drop=True)
        self.df_index["length"] = self.df_index["target_phoneme"].apply(lambda x: len(x))
        self.df_index = self.df_index.sort_values(by="length", ascending=True).reset_index(drop=True)
        
        with open(file_vocab, "r") as f_vocab:
            self.dict_vocab = json.load(f_vocab)
        
        self.path_sample = os.path.join(path_sample, split)
    
    def __getitem__(self, index):
        file_sample = os.path.join(self.path_sample, "{}.npy".format(str(self.df_index["sample_index"].values[index]).zfill(4)))
        array_feature = np.load(file_sample).reshape(-1)
        
        target_text = self.df_index["target_phoneme"].values[index][2: -2].split("', '")
        label = np.array(list(map(lambda x: self.dict_vocab[x], target_text)))
        
        return {"feature": array_feature, "label": label}
    
    def __len__(self):
        return len(self.df_index)
    
    def collate(self, batch):
        tensor_feature = pad_sequence(list(map(lambda x: torch.from_numpy(x["feature"]), batch)), batch_first=True)
        tensor_label = pad_sequence(list(map(lambda x: torch.from_numpy(x["label"]), batch)), batch_first=True, padding_value=self.dict_vocab[PAD])
        
        return {"feature": tensor_feature, "label": tensor_label}


def test():
    """
    
    """
    
    file_index = "/Users/nanzheng/Desktop/wav2vec_2.0/data/timit/index.txt"
    file_phoneme_dict = "/Users/nanzheng/Desktop/wav2vec_2.0/model/phoneme_dict.json"
    path_sample = "/Users/nanzheng/Desktop/wav2vec_2.0/data/timit"
    
    # dataset = DatasetWav2Vec2Phn(file_index, file_phoneme_dict, path_sample)
    # data_loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=dataset.collate)
    # for i, batch in enumerate(data_loader):
    #     print(batch["feature"].shape)
    #     print(batch["label"].shape)
    
    dataset = DatasetSeq2SeqPhn(file_index, file_phoneme_dict, path_sample)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=dataset.collate)
    for i, batch in enumerate(data_loader):
        print(batch["feature"].shape)
        print(batch["label"].shape)
    
    return None


def test_raw_phn():
    """
    
    """
    
    file_index = "/Users/nanzheng/Desktop/wav2vec_2.0/data/timit/index_raw.txt"
    file_phoneme_dict = "/Users/nanzheng/Desktop/wav2vec_2.0/model/phoneme_dict.json"
    path_sample = "/Users/nanzheng/Desktop/wav2vec_2.0/data/timit_raw"
    
    dataset = DatasetRawWav2Vec2Phn(file_index, file_phoneme_dict, path_sample)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=dataset.collate)
    for i, batch in enumerate(data_loader):
        print(batch["feature"].shape)
        print(batch["label"].shape)
    
    return None


if __name__ == "__main__":
    test_raw_phn()
    
    