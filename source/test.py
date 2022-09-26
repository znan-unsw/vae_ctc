#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 15:42:05 2022

@author: nanzheng
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
import argparse
import configparser

from train import preprocess, validate
from utils import get_dataset, get_model_config, get_model_and_loss, iterate, get_latest_checkpoint


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("Using {} now.".format(DEVICE))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_config", type=str, default=None)
    args, _ = parser.parse_known_args()
    
    file_config = args.file_config
    # file_config = "../config/config_ctc.ini"
    # file_config = "../config/config_seq2seq.ini"
    # file_config = "../config/config_vae_ctc.ini"
    # file_config = "../config/config_dependent_vae_ctc.ini"
    
    config = configparser.ConfigParser()
    config.read(file_config)
    
    file_index = config.get("path", "file_index")
    file_vocab = config.get("path", "file_vocab")
    path_sample = config.get("path", "path_sample")
    path_checkpoint = config.get("path", "path_checkpoint")
    
    with open(file_vocab, "r") as f_vocab:
        dict_vocab = json.load(f_vocab)
    
    dict_train_args = {
        "device": DEVICE,
        
        "start_iter": config.getint("training", "start_iter"),    # initial setting, might be varied if checkpoint resumed
        "start_epoch": config.getint("training", "start_epoch"),    # initial setting, might be varied if checkpoint resumed
        "n_epoch": config.getint("training", "n_epoch"),
        "batch_size": config.getint("training", "batch_size"),
        "learning_rate": config.getfloat("training", "learning_rate"),
        "iter_valid": config.getint("training", "iter_valid"),    # iteration
        "iter_checkpoint": config.getint("training", "iter_checkpoint"),    # epoch
        "model_ver": config.get("training", "model_ver"),
        
        "file_index": file_index,
        "file_vocab": file_vocab,
        "path_sample": path_sample,
        "dict_vocab": dict_vocab,
        "dict_vocab_re": dict(zip(list(dict_vocab.values()), list(dict_vocab.keys())))
        }
    
    load_checkpoint = bool(config.getboolean("training", "load_checkpoint"))
    if load_checkpoint:
        try:
            file_latest_checkpoint = get_latest_checkpoint(path_checkpoint, dict_train_args["model_ver"])
            print("Resuming from checkpoint {} ...".format(file_latest_checkpoint))
        except Exception:
            file_latest_checkpoint = None
    else:
        file_latest_checkpoint = None
    
    dict_model_config = get_model_config(dict_train_args, config)
    model, loss_func, opt, data_loader_train, data_loader_valid = preprocess(dict_train_args, dict_model_config, file_latest_checkpoint)
    avg_loss, avg_per = validate(model, loss_func, data_loader_valid, dict_train_args, quiet=False)
    
    print("Phoneme error rate:")
    print("\t{:.4f}".format(avg_per))
    
    