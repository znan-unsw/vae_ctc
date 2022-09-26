#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 09:23:47 2022

@author: nanzheng
"""

import torch
import torch.nn.functional as F
from torchsummary import summary

from seq_to_seq import EncoderDecoder


class ModelCtc(torch.nn.Module):
    """
    
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.linear = torch.nn.Linear(config["dim_input"], config["dim_output"])
    
    def forward(self, X):
        return self.linear(X)


class ModelVaeCtc(torch.nn.Module):
    """
    
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.device = config["device"]
        
        dim_input = config["dim_input"]
        self.dim_vae_hidden = config["vae->dim_vae_hidden"]
        dim_output = config["dim_output"]
        
        self.get_p_mu = torch.nn.Sequential(
            torch.nn.Linear(dim_input, self.dim_vae_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dim_vae_hidden, self.dim_vae_hidden),
            )
        self.get_p_sigma = torch.nn.Sequential(
            torch.nn.Linear(dim_input, self.dim_vae_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dim_vae_hidden, self.dim_vae_hidden),
            )
        self.get_q_mu = torch.nn.Sequential(
            torch.nn.Linear(dim_input, self.dim_vae_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dim_vae_hidden, self.dim_vae_hidden),
            )
        self.get_q_sigma = torch.nn.Sequential(
            torch.nn.Linear(dim_input, self.dim_vae_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dim_vae_hidden, self.dim_vae_hidden),
            )
        
        self.linear_z = torch.nn.Linear(self.dim_vae_hidden, self.dim_vae_hidden)
        self.hidden = torch.nn.Linear(dim_input + self.dim_vae_hidden, self.dim_vae_hidden)
        self.output = torch.nn.Linear(self.dim_vae_hidden, dim_output)
    
    def forward(self, X):
        mu_p = self.get_p_mu(X)
        sigma_p = self.get_p_sigma(X)   # log(sigma_p^2)
        mu_q = self.get_q_mu(X)
        sigma_q = self.get_q_sigma(X)   # log(sigma_q^2)
        
        epsilon = torch.randn(self.dim_vae_hidden).to(self.device)
        
        Z = mu_q + torch.exp(sigma_q / 2) * epsilon
        
        tilde_Z = F.relu(self.linear_z(Z))
        tilde_X = torch.cat((X, tilde_Z), 2)
        tilde_X = F.relu(self.hidden(tilde_X))
        
        return self.output(tilde_X), mu_p, sigma_p, mu_q, sigma_q


class ModelDependentVaeCtc(torch.nn.Module):
    """
    
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.device = config["device"]
        
        dim_input = config["dim_input"]
        self.dim_vae_hidden = config["vae->dim_vae_hidden"]
        dim_output = config["dim_output"]
        
        self.get_p_mu = torch.nn.GRU(dim_input, int(self.dim_vae_hidden / 2), batch_first=True, bidirectional=True)
        self.get_p_sigma = torch.nn.GRU(dim_input, int(self.dim_vae_hidden / 2), batch_first=True, bidirectional=True)
        self.get_q_mu = torch.nn.Sequential(
            torch.nn.Linear(dim_input, self.dim_vae_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dim_vae_hidden, self.dim_vae_hidden),
            )
        self.get_q_sigma = torch.nn.Sequential(
            torch.nn.Linear(dim_input, self.dim_vae_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dim_vae_hidden, self.dim_vae_hidden),
            )
        
        self.linear_z = torch.nn.Linear(self.dim_vae_hidden, self.dim_vae_hidden)
        self.hidden = torch.nn.Linear(dim_input + self.dim_vae_hidden, self.dim_vae_hidden)
        self.output = torch.nn.Linear(self.dim_vae_hidden, dim_output)
        
    def forward(self, X):
        mu_p, _ = self.get_p_mu(X)
        sigma_p, _ = self.get_p_sigma(X)   # log(sigma_p^2)
        mu_q = self.get_q_mu(X)
        sigma_q = self.get_q_sigma(X)   # log(sigma_q^2)
        
        epsilon = torch.randn(self.dim_vae_hidden).to(self.device)
        
        Z = mu_q + torch.exp(sigma_q / 2) * epsilon
        
        tilde_Z = F.relu(self.linear_z(Z))
        tilde_X = torch.cat((X, tilde_Z), 2)
        tilde_X = F.relu(self.hidden(tilde_X))
        
        return self.output(tilde_X), mu_p, sigma_p, mu_q, sigma_q


class ModelMarkovVaeCtc(torch.nn.Module):
    """
    
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.device = config["device"]
        
        dim_input = config["dim_input"]
        self.dim_vae_hidden = config["vae->dim_vae_hidden"]
        dim_output = config["dim_output"]
        
        self.get_p_mu_0 = torch.nn.Sequential(
            torch.nn.Linear(dim_input, self.dim_vae_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dim_vae_hidden, self.dim_vae_hidden),
            )
        self.get_p_sigma_0 = torch.nn.Sequential(
            torch.nn.Linear(dim_input, self.dim_vae_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dim_vae_hidden, self.dim_vae_hidden),
            )
        self.get_p_mu = torch.nn.Sequential(
            torch.nn.Linear(dim_input + self.dim_vae_hidden * 2, self.dim_vae_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dim_vae_hidden, self.dim_vae_hidden),
            )
        self.get_p_sigma = torch.nn.Sequential(
            torch.nn.Linear(dim_input + self.dim_vae_hidden * 2, self.dim_vae_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dim_vae_hidden, self.dim_vae_hidden),
            )
        self.get_q_mu = torch.nn.Sequential(
            torch.nn.Linear(dim_input, self.dim_vae_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dim_vae_hidden, self.dim_vae_hidden),
            )
        self.get_q_sigma = torch.nn.Sequential(
            torch.nn.Linear(dim_input, self.dim_vae_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dim_vae_hidden, self.dim_vae_hidden),
            )
        
        self.linear_z = torch.nn.Linear(self.dim_vae_hidden, self.dim_vae_hidden)
        self.hidden = torch.nn.Linear(dim_input + self.dim_vae_hidden, self.dim_vae_hidden)
        self.output = torch.nn.Linear(self.dim_vae_hidden, dim_output)
    
    def forward(self, X):
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        
        mu_p = torch.zeros(batch_size, seq_len, self.dim_vae_hidden).to(self.device)
        sigma_p = torch.zeros(batch_size, seq_len, self.dim_vae_hidden).to(self.device)  # log(sigma_q^2)
        mu_q = self.get_q_mu(X)
        sigma_q = self.get_q_sigma(X)   # log(sigma_q^2)
        
        epsilon = torch.randn(self.dim_vae_hidden).to(self.device)
        
        Z = mu_q + torch.exp(sigma_q / 2) * epsilon
        
        mu_p[:, 0, :] = self.get_p_mu_0(X[:, 0, :])
        sigma_p[:, 0, :] = self.get_p_sigma_0(X[:, 0, :])
        for t in range(1, seq_len):
            tilde_x_t = torch.cat((mu_p[:, t - 1, :], sigma_p[:, t - 1, :], X[:, t, :]), dim=-1)
            mu_p[:, t, :] = self.get_p_mu(tilde_x_t)
            sigma_p[:, t, :] = self.get_p_sigma(tilde_x_t)
        
        tilde_Z = F.relu(self.linear_z(Z))
        tilde_X = torch.cat((X, tilde_Z), 2)
        tilde_X = F.relu(self.hidden(tilde_X))
        
        return self.output(tilde_X), mu_p, sigma_p, mu_q, sigma_q


class ModelSeqToSeq(EncoderDecoder):
    """
    
    """
    
    def __init__(self, config):
        super().__init__(config)


def test():
    """
    
    """
    
    # config = {
    #     "dim_input": 128,
    #     "dim_output": 64
    #     }
    # model = ModelCtc(config)
    
    # config = {
    #     "device": "cpu",
    #     "encoder->dim_encoder_input": 768,
    #     "encoder->dim_encoder_hidden": 256,
    #     "decoder->dim_decoder_hidden": 256,
    #     "decoder->dim_embedding": 8,
    #     "decoder->p_dropout": 0.5,
    #     "decoder->p_teacher_force": 0.5,
    #     "decoder->dim_output": 62
    #     }
    # model = ModelSeqToSeq(config)
    
    # config = {
    #     "device": "cpu",
    #     "dim_input": 768,
    #     "vae->dim_vae_hidden": 256,
    #     "dim_output": 64
    #     }
    # model = ModelVaeCtc(config)
    # summary(model, input_size=(10, 768))
    
    config = {
        "device": "cpu",
        "dim_input": 768,
        "vae->dim_vae_hidden": 256,
        "dim_output": 64
        }
    model = ModelDependentVaeCtc(config)
    summary(model, input_size=(10, 768))
    
    # config = {
    #     "device": "cpu",
    #     "dim_input": 768,
    #     "vae->dim_vae_hidden": 256,
    #     "dim_output": 64
    #     }
    # model = ModelMarkovVaeCtc(config)
    # summary(model, input_size=(10, 768))
    
    return None


if __name__ == "__main__":
    test()
    
    