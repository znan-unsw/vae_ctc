#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 09:36:50 2022

@author: nanzheng
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
import argparse
import configparser

from utils import get_dataset, get_model_config, get_model_and_loss, iterate, get_latest_checkpoint


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} now.".format(DEVICE))


def validate(model, loss_func, data_loader_valid, dict_train_args, quiet=False):
    """
    
    """
    
    model.eval()
    
    batch_size = dict_train_args["batch_size"]
    total_batch = int(len(data_loader_valid.dataset) / batch_size)
    
    loss_valid = []
    list_per_valid = []
    for idx_batch, batch in enumerate(data_loader_valid):
        if not quiet:
            print("Validating Batch: {}/{}".format(idx_batch, total_batch))
        loss, per = iterate(model, batch, loss_func, dict_train_args, mode="test")
        
        loss_valid.append(loss.item())
        list_per_valid.append(per)
    
    avg_loss = np.mean(loss_valid)
    avg_per = np.mean(list_per_valid)
    
    model.train()
    
    return avg_loss, avg_per


def train(model, loss_func, opt, scheduler, data_loader_train, data_loader_valid, dict_train_args):
    """
    
    """
    
    model.train()
    
    start_epoch = dict_train_args["start_epoch"]
    n_epoch = dict_train_args["n_epoch"]
    batch_size = dict_train_args["batch_size"]
    iter_valid = dict_train_args["iter_valid"]
    iter_checkpoint = dict_train_args["iter_checkpoint"]
    idx_iter = dict_train_args["start_iter"]
    model_ver = dict_train_args["model_ver"]
    total_batch = int(len(data_loader_train.dataset) / batch_size)
    
    try:
        list_loss_train = dict_train_args["result"]["train_loss"]
        list_loss_valid = dict_train_args["result"]["valid_loss"]
        list_valid_iter = dict_train_args["result"]["valid_iter"]
        
        list_per_train = dict_train_args["result"]["train_per"]
        list_per_valid = dict_train_args["result"]["valid_per"]
        
    except Exception:
        list_loss_train = []
        list_loss_valid = []
        list_valid_iter = []
        
        list_per_train = []
        list_per_valid = []
    
    for idx_epoch in range(start_epoch, start_epoch + n_epoch):
        for idx_batch, batch in enumerate(data_loader_train):
            loss, per = iterate(model, batch, loss_func, dict_train_args)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            list_loss_train.append(loss.item())
            list_per_train.append(per)
            
            print("Epoch: {}/{}, Batch: {}/{}, train_loss: {:.4f}, train_per: {:.4f}".format( \
                    idx_epoch, start_epoch + n_epoch, idx_batch, total_batch, 
                    list_loss_train[-1], list_per_train[-1]))
            
            if (idx_iter != 0) and (idx_iter % iter_valid == 0):
                loss, per = validate(model, loss_func, data_loader_valid, dict_train_args)
                list_loss_valid.append(loss)
                list_per_valid.append(per)
                list_valid_iter.append(idx_iter)
                
                print("Epoch: {}/{}, Batch: {}/{}, train_loss: {:.4f}, train_per: {:.4f}, valid_loss: {:.4f}, valid_per: {:.4f}".format( \
                        idx_epoch, start_epoch + n_epoch, idx_batch, total_batch, 
                        list_loss_train[-1], list_per_train[-1], list_loss_valid[-1], list_per_valid[-1]))
                
            idx_iter += 1
        scheduler.step()
        
        if idx_epoch % iter_checkpoint == 0:
            dict_result = {
                "train_loss": list_loss_train,
                "valid_loss": list_loss_valid,
                "train_per": list_per_train, 
                "valid_per": list_per_valid,
                "valid_iter": list_valid_iter
                }
            
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": idx_epoch,
                "iter": idx_iter,
                "result": dict_result
                }
            file_checkpoint = os.path.join(path_checkpoint, "checkpoint_{}_epoch_{}.pkl".format(model_ver, str(idx_epoch).zfill(3)))
            torch.save(checkpoint, file_checkpoint)
            print("{} saved.".format(file_checkpoint))
        
    dict_result = {
        "train_loss": list_loss_train,
        "valid_loss": list_loss_valid,
        "train_per": list_per_train, 
        "valid_per": list_per_valid,
        "valid_iter": list_valid_iter
        }
    
    return dict_result


def preprocess(dict_train_args, dict_model_config, file_latest_checkpoint=None, test=False):
    """

    """
    
    batch_size = dict_train_args["batch_size"]
    learning_rate = dict_train_args["learning_rate"]
    
    file_index = dict_train_args["file_index"]
    file_vocab = dict_train_args["file_vocab"]
    path_sample = dict_train_args["path_sample"]
    
    Dataset = get_dataset(dict_train_args)
    model, loss_func = get_model_and_loss(dict_train_args, dict_model_config)
    # model = model.double()
    model = model.to(DEVICE)
    
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.1)
    
    if file_latest_checkpoint:
        if torch.cuda.is_available():
            checkpoint = torch.load(file_latest_checkpoint)
        else:
            checkpoint = torch.load(file_latest_checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.last_epoch = checkpoint["epoch"]
        dict_train_args["start_epoch"] = checkpoint["epoch"] + 1
        dict_train_args["start_iter"] = checkpoint["iter"]
        dict_train_args["result"] = checkpoint["result"]
    
    if test == False:
        dataset_train = Dataset(file_index, file_vocab, path_sample, split="train")
        dataset_valid = Dataset(file_index, file_vocab, path_sample, split="test")
        
        data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=dataset_train.collate)
        data_loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, collate_fn=dataset_valid.collate)
        
        return model, loss_func, opt, scheduler, data_loader_train, data_loader_valid
    
    else:
        dataset_valid = Dataset(file_index, file_vocab, path_sample, split="test")
        data_loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, collate_fn=dataset_valid.collate)
        
        return model, data_loader_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_config", type=str, default=None)
    args, _ = parser.parse_known_args()
    
    file_config = args.file_config
    # file_config = "../config/config_ctc.ini"
    # file_config = "../config/config_vae_ctc.ini"
    # file_config = "../config/config_dependent_vae_ctc.ini"
    # file_config = "../config/config_markov_vae_ctc.ini"
    
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
    model, loss_func, opt, scheduler, data_loader_train, data_loader_valid = preprocess(dict_train_args, dict_model_config, file_latest_checkpoint)
    train(model, loss_func, opt, scheduler, data_loader_train, data_loader_valid, dict_train_args)
    
    