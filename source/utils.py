#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:53:36 2022

@author: nanzheng
"""

import os
import numpy as np
import torch
import editdistance
from operator import itemgetter
from itertools import groupby
import subprocess
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, \
        Wav2Vec2ForCTC, TrainingArguments, Trainer

from dataset import DatasetWav2Vec2Phn, DatasetSeq2SeqPhn, DatasetRawWav2Vec2Phn
from model import ModelCtc, ModelVaeCtc, ModelDependentVaeCtc, ModelMarkovVaeCtc, ModelSeqToSeq
from constants import from_60_to_39_phn, SOS, EOS, PAD


def get_model_config(dict_train_args, config, raw=False):
    """
    
    """
    
    if dict_train_args["model_ver"] == "ctc":
        dict_model_config = {
            "dim_input": config.getint("model", "dim_input"),
            "dim_output": len(dict_train_args["dict_vocab"])
            }
    
    elif dict_train_args["model_ver"] in ["vae_ctc", "dependent_vae_ctc", "markov_vae_ctc"]:
        dict_model_config = {
            "device": dict_train_args["device"],
            "dim_input": config.getint("model", "dim_input"),
            "vae->dim_vae_hidden": config.getint("model", "vae->dim_vae_hidden"),
            "dim_output": len(dict_train_args["dict_vocab"])
            }
    
    elif dict_train_args["model_ver"] == "seq2seq":
        # del dict_train_args["dict_vocab_re"][dict_train_args["dict_vocab"][PAD]]
        # del dict_train_args["dict_vocab"][PAD]
        dict_train_args["dict_vocab"][EOS] = len(dict_train_args["dict_vocab"])
        dict_train_args["dict_vocab_re"][len(dict_train_args["dict_vocab_re"])] = EOS
        
        dict_model_config = {
            "device": dict_train_args["device"],
            "encoder->bidirectional": config.getboolean("model", "encoder->bidirectional"),
            "encoder->dim_encoder_input": config.getint("model", "encoder->dim_encoder_input"),
            "encoder->dim_encoder_hidden": config.getint("model", "encoder->dim_encoder_hidden"),
            "decoder->dim_decoder_hidden": config.getint("model", "decoder->dim_decoder_hidden"),
            "decoder->dim_embedding": config.getint("model", "decoder->dim_embedding"),
            "decoder->p_dropout": config.getfloat("model", "decoder->p_dropout"),
            "decoder->p_teacher_force": config.getfloat("model", "decoder->p_teacher_force"),
            "decoder->dim_output": len(dict_train_args["dict_vocab"])
            }
    
    if raw:
        dict_model_config["pretrained_wav2vec2"] = config.get("model", "pretrained_wav2vec2")
        dict_model_config["model_ver"] = dict_train_args["model_ver"]
    
    return dict_model_config


def get_dataset(dict_train_args, raw=False):
    """
    
    """
    
    if dict_train_args["model_ver"] in ["ctc", "vae_ctc", "dependent_vae_ctc", "markov_vae_ctc"]:
        if raw:
            Dataset = DatasetRawWav2Vec2Phn
        else:
            Dataset = DatasetWav2Vec2Phn
    elif dict_train_args["model_ver"] == "seq2seq":
        Dataset = DatasetSeq2SeqPhn
    
    return Dataset


def _kl_gaussian(mu_1, sigma_1, mu_2, sigma_2):
    """
    
    Parameters
    ----------
    sigma_1: log(sigma_1^2)
    sigma_2: log(sigma_2^2)
    """
    
    kl = 0.5 * (sigma_2 - sigma_1 + torch.exp(sigma_1 - sigma_2) + torch.pow(mu_1 - mu_2, 2) / torch.exp(sigma_2) - 1)
    kl = torch.where(torch.isinf(kl), torch.full_like(kl, 100), kl)
    
    return kl


def _get_model_and_loss_finetune(dict_train_args, dict_model_config):
    """
    
    """
    processor = Wav2Vec2Processor.from_pretrained(dict_model_config["pretrained_wav2vec2"])
    model = Wav2Vec2ForCTC.from_pretrained(dict_model_config["pretrained_wav2vec2"], 
                                           gradient_checkpointing=True, 
                                           ctc_loss_reduction="mean", 
                                           pad_token_id=processor.tokenizer.pad_token_id,
                                           vocab_size=len(dict_train_args["dict_vocab"]),
                                           dict_model_config=dict_model_config
                                           )
    
    if dict_train_args["model_ver"] == "ctc":
        label_padding = dict_train_args["dict_vocab"][PAD]
        loss_func = torch.nn.CTCLoss(blank=label_padding, zero_infinity=False)
    elif dict_train_args["model_ver"] == "vae_ctc":
        label_padding = dict_train_args["dict_vocab"][PAD]
        loss_func = {
            "ctc": torch.nn.CTCLoss(blank=label_padding, zero_infinity=False),
            "kl": _kl_gaussian
            }
    
    return model, loss_func


def get_model_and_loss(dict_train_args, dict_model_config, raw=False):
    """
    
    """
    
    if raw:
        return _get_model_and_loss_finetune(dict_train_args, dict_model_config)
    
    if dict_train_args["model_ver"] == "ctc":
        model = ModelCtc(dict_model_config)
        
        label_padding = dict_train_args["dict_vocab"][PAD]
        loss_func = torch.nn.CTCLoss(blank=label_padding, zero_infinity=False)
    
    elif dict_train_args["model_ver"] == "vae_ctc":
        model = ModelVaeCtc(dict_model_config)
        
        label_padding = dict_train_args["dict_vocab"][PAD]
        loss_func = {
            "ctc": torch.nn.CTCLoss(blank=label_padding, zero_infinity=False),
            "kl": _kl_gaussian
            }
    
    elif dict_train_args["model_ver"] == "dependent_vae_ctc":
        model = ModelDependentVaeCtc(dict_model_config)
        
        label_padding = dict_train_args["dict_vocab"][PAD]
        loss_func = {
            "ctc": torch.nn.CTCLoss(blank=label_padding, zero_infinity=False),
            "kl": _kl_gaussian
            }
    
    elif dict_train_args["model_ver"] == "markov_vae_ctc":
        model = ModelMarkovVaeCtc(dict_model_config)
        
        label_padding = dict_train_args["dict_vocab"][PAD]
        loss_func = {
            "ctc": torch.nn.CTCLoss(blank=label_padding, zero_infinity=False),
            "kl": _kl_gaussian
            }
        
    elif dict_train_args["model_ver"] == "seq2seq":
        model = ModelSeqToSeq(dict_model_config)
        
        label_padding = dict_train_args["dict_vocab"]["[PAD]"]
        # label_padding = dict_train_args["dict_vocab"][EOS]
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=label_padding)
    
    return model, loss_func


def iterate(model, batch, loss_func, dict_train_args, mode="train"):
    """
    
    """
    
    device = dict_train_args["device"]
    
    X = batch["feature"].to(device)
    y = batch["label"].to(device)
    
    if dict_train_args["model_ver"] == "ctc":
        pad = PAD
        label_padding = dict_train_args["dict_vocab"][PAD]
        
        y_pred = model(X)
    
        target_lengths = (y != label_padding).long().sum(dim=-1)
        trimmed_targets = [t[: l] for t, l in zip(y, target_lengths)]
        targets = torch.cat(trimmed_targets)
        log_probs = torch.nn.functional.log_softmax(y_pred, dim=-1, dtype=torch.float32).transpose(0, 1)
        input_lengths = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.long)
        
        loss = loss_func(log_probs, targets, input_lengths, target_lengths)
    
    elif dict_train_args["model_ver"] in ["vae_ctc", "dependent_vae_ctc", "markov_vae_ctc"]:
        pad = PAD
        label_padding = dict_train_args["dict_vocab"][PAD]
        
        y_pred, mu_p, sigma_p, mu_q, sigma_q = model(X)
        
        target_lengths = (y != label_padding).long().sum(dim=-1)
        trimmed_targets = [t[: l] for t, l in zip(y, target_lengths)]
        targets = torch.cat(trimmed_targets)
        log_probs = torch.nn.functional.log_softmax(y_pred, dim=-1, dtype=torch.float32).transpose(0, 1)
        input_lengths = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.long)
        
        loss_func_ctc = loss_func["ctc"]
        loss_func_kl = loss_func["kl"]
        
        ctc = loss_func_ctc(log_probs, targets, input_lengths, target_lengths)
        ctc = torch.where(torch.isinf(ctc), torch.full_like(ctc, 100), ctc)
        loss = ctc + torch.mean(loss_func_kl(mu_q, sigma_q, mu_p, sigma_p))
        
    elif dict_train_args["model_ver"] == "seq2seq":
        # pad = EOS
        pad = PAD
        
        X = X.transpose(0, 1)
        y_ = y.transpose(0, 1)
        
        y_pred = model(X, y_, mode=mode)
        y_pred[0, :, dict_train_args["dict_vocab"][SOS]] = 1
        
        log_probs = torch.nn.functional.log_softmax(y_pred, dim=-1, dtype=torch.float32)
        
        dim_output = y_pred.shape[-1]
        y_ = y_[1: , :].reshape(-1)
        y_pred = y_pred[1: , :, :].reshape(-1, dim_output)
        
        loss = loss_func(y_pred, y_)
        
    list_pred = decode(log_probs, dict_train_args["dict_vocab_re"], dict_train_args["model_ver"])
    per = calc_per(list_pred, y, dict_train_args["dict_vocab_re"], pad=pad)
    
    return loss, per


def decode(log_probs, dict_phoneme_re, model_ver):
    """
    
    """
    
    list_pred_phn = []
    if torch.cuda.is_available():
        array_idx_pred = torch.argmax(log_probs, dim=-1).cpu().detach().numpy()
    else:
        array_idx_pred = torch.argmax(log_probs, dim=-1).detach().numpy()
    for i in range(array_idx_pred.shape[1]):
        list_pred_phn_ = list(map(lambda x: dict_phoneme_re[x], array_idx_pred[:, i]))
        if model_ver == "seq2seq":
            pass
        else:
            list_pred_phn_ = list(map(itemgetter(0), groupby(list_pred_phn_)))
        list_pred_phn.append(list_pred_phn_)
        
    return list_pred_phn


def _60_to_39(list_phn):
    """

    """
    
    return list(map(lambda x: from_60_to_39_phn[x], list_phn))


def calc_per(list_pred_phn, y_true, dict_phoneme_re, pad=PAD):
    """
    
    """
    
    list_per = []
    for i in range(len(list_pred_phn)):
        list_pred_phn_ = list_pred_phn[i]
        while pad in list_pred_phn_:
            list_pred_phn_.remove(pad)
        list_pred_phn_ = _60_to_39(list_pred_phn_)
        
        if torch.cuda.is_available():
            y_true_ = y_true[i].cpu().detach().numpy()
        else:
            y_true_ = y_true[i].detach().numpy()
        y_true_ = list(map(lambda x: dict_phoneme_re[x], y_true_))
        while pad in y_true_:
            y_true_.remove(pad)
        y_true_ = _60_to_39(y_true_)
        
        edit_distance = editdistance.eval(y_true_, list_pred_phn_)
        per = edit_distance / len(y_true_)
        list_per.append(per)
    
    return np.mean(list_per)


def get_latest_checkpoint(path_checkpoint, model_ver):
    """
    FIXME: -
    """
    
    ret = subprocess.getoutput("ls -al {}".format(path_checkpoint))
    ret = ret.split("\n")
    list_line = []
    for line in ret:
        if "checkpoint_{}".format(model_ver) in line:
            list_line.append(line)
    
    if len(list_line) == 0:
        raise FileNotFoundError()
    
    list_create_time = list(map(lambda x: x.split()[-4: -1], list_line))
    list_create_time = list(map(lambda x: x[-3: -2] + [x[-2].zfill(2)] + x[-1: ], list_create_time))
    list_create_time = list(map(lambda x: "_".join(x), list_create_time))
    
    list_checkpoint = list(map(lambda x: x.split(" ")[-1], list_line))
    list_checkpoint = list(zip(list_create_time, list_checkpoint))
    list_checkpoint = sorted(list_checkpoint, key=lambda x: x[1], reverse=True)
    list_checkpoint = sorted(list_checkpoint, key=lambda x: x[0], reverse=True)

    latest_checkpoint = list_checkpoint[0][-1]
    
    file_latest_checkpoint = os.path.join(path_checkpoint, latest_checkpoint)
    
    return file_latest_checkpoint

