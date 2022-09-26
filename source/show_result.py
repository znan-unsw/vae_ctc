#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:36:24 2022

@author: nanzheng
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import configparser

from dataset import DatasetSeq2SeqPhn
from utils import get_latest_checkpoint


list_color = ["b", "k", "r", "c", "y", "m", "g"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_config", type=str, default=None)
    args, _ = parser.parse_known_args()
    
    list_file_config = [
        "../config/config_ctc.ini",
        # "../config/config_vae_ctc.ini",
        # "../config/config_markov_vae_ctc.ini",
        # "../config/config_dependent_vae_ctc.ini"
        ]

    list_checkpoint = []
    list_model_ver = []
    for file_config in list_file_config:
        config = configparser.ConfigParser()
        config.read(file_config)
        
        file_index = config.get("path", "file_index")
        file_vocab = config.get("path", "file_vocab")
        path_sample = config.get("path", "path_sample")
        path_checkpoint = config.get("path", "path_checkpoint")
        batch_size = config.getint("training", "batch_size")
        
        file_latest_checkpoint = get_latest_checkpoint(path_checkpoint, config.get("training", "model_ver"))
        list_checkpoint.append(file_latest_checkpoint)
        list_model_ver.append(config.get("training", "model_ver"))
    
    plt.figure()
    for n, model in enumerate(list_checkpoint):
        file_latest_checkpoint = list_checkpoint[n]
        print("Resuming from checkpoint {} ...".format(file_latest_checkpoint))
        checkpoint = torch.load(file_latest_checkpoint, map_location=torch.device("cpu"))
        dict_result = checkpoint["result"]
        
        dataset_train = DatasetSeq2SeqPhn(file_index, file_vocab, path_sample, split="train")
        num_train = len(dataset_train)
        
        x = np.array([i for i in range(len(dict_result["train_per"]))]) * batch_size / num_train
        y = np.array(dict_result["train_per"])
        x_new = []
        y_new = []
        idx_epoch = 1
        i_start = 0
        for i in range(x.shape[0]):
            if x[i] > idx_epoch:
                x_new.append(x[i])
                y_new.append(np.mean(y[i_start: i - 1]))
                idx_epoch += 1
                i_start = i
            
            if idx_epoch >= 1110:
                break
        
        x_new_ = []
        y_new_ = []
        window_len = 2
        for i in range(len(x_new)):
            x_new_.append(np.mean(x_new[max(0, i - window_len): min(i + window_len + 1, len(x_new))]))
            y_new_.append(np.mean(y_new[max(0, i - window_len): min(i + window_len + 1, len(y_new))]))
            
        plt.plot(x_new, y_new, "{}-".format(list_color[n % len(list_color)]), linewidth=1.0, label="dev - {}".format(list_model_ver[n]))
        print("{}: {:.4f}".format(list_model_ver[n], y_new[-1]))
        
        x = np.array(dict_result["valid_iter"]) * batch_size / num_train
        y = np.array(dict_result["valid_per"])
        x_new = []
        y_new = []
        if x[1] - x[0] > 1:
            i = 0
            while i < x.shape[0] and x[i] <= 1110:
                x_new.append(x[i])
                y_new.append(y[i])
                i += 1
        else:
            idx_epoch = 1
            i_start = 0
            for i in range(x.shape[0]):
                print(idx_epoch)
                if x[i] > idx_epoch:
                    x_new.append(x[i])
                    y_new.append(y[i - 1])
                    idx_epoch += 1
                    i_start = i
        
        x_new_ = []
        y_new_ = []
        window_len = 2
        for i in range(len(x_new)):
            x_new_.append(np.mean(x_new[max(0, i - window_len): min(i + window_len + 1, len(x_new))]))
            y_new_.append(np.mean(y_new[max(0, i - window_len): min(i + window_len + 1, len(y_new))]))
        
        plt.plot(x_new, y_new, "{}--".format(list_color[n % len(list_color)]), linewidth=2.0, label="test - {}".format(list_model_ver[n]))
        # plt.plot(x_new, [y_new[-1]] * len(x_new), "{}--".format(list_color[n % len(list_color)]), linewidth=0.8, alpha=0.6)
        print("{}: {:.4f}".format(list_model_ver[n], y_new[-1]))
    
    plt.legend(fontsize=10)
    plt.grid()
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("Epoch Index", fontsize=10)
    plt.ylabel("PER", fontsize=10)
    plt.ylim([0, 1])
    # plt.savefig("fig.pdf", format="pdf")
    plt.show()
    
    
    # plt.figure()
    # for n, model in enumerate(list_checkpoint):
    #     file_latest_checkpoint = list_checkpoint[n]
    #     checkpoint = torch.load(file_latest_checkpoint, map_location=torch.device("cpu"))
    #     dict_result = checkpoint["result"]
        
    #     plt.plot(dict_result["train_loss"], linewidth=0.8, alpha=0.5, label="{}".format(model.split("/")[-1]))
    # plt.legend()
    # plt.grid()
    # plt.xlabel("Iteration Index (batch_size = {})".format(batch_size))
    # plt.ylabel("Training Loss")
    # plt.show()
    
    