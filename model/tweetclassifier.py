import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

import numpy as np

from pdb import set_trace

'''
breach: Databreach
fp: Fraud/Phishing
rm: Ransomware/Malware
ddos: DDoS
vuln: Vulnerability
uninfo: Uninformative
nonsec: Non-security 
'''

class TweetClassifier(nn.Module):
    def __init__(self, base_model, hidden_dim=768):
        super().__init__()

        self.base_model = base_model

        if base_model == "bert-base-cased":
            print('load bert base cased model')            
            self.lm = AutoModel.from_pretrained('bert-base-cased')
        elif base_model == "bert-base-uncased":
            print('load bert base uncased model')            
            self.lm = AutoModel.from_pretrained('bert-base-uncased')            
        elif base_model  == "roberta-base":
            print('load roberta base model')            
            self.lm = AutoModel.from_pretrained('roberta-base')
        elif base_model  == "securebert":
            print('load secrebert model')            
            self.lm = AutoModel.from_pretrained('ehsanaghaei/SecureBERT')            
        elif base_model == 'bertweet':
            print('load bertweet model')
            self.lm  = AutoModel.from_pretrained('vinai/bertweet-base')

        self.dp_cls = nn.Linear(hidden_dim, 2)
        self.fp_cls = nn.Linear(hidden_dim, 2)
        self.rm_cls = nn.Linear(hidden_dim, 2)
        self.ddos_cls = nn.Linear(hidden_dim, 2)
        self.vuln_cls = nn.Linear(hidden_dim, 2)
        self.uninfo_cls = nn.Linear(hidden_dim, 2)
        self.nonsec_cls = nn.Linear(hidden_dim, 2)

    def forward(self, input_ids, attention_mask):
        lm_out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        lm_cls_h = lm_out[0][:, 0, :]

        # Feed input to classifiers to compute logits
        dp_logits = self.dp_cls(lm_cls_h)
        fp_logits = self.fp_cls(lm_cls_h)
        rm_logits = self.rm_cls(lm_cls_h)
        ddos_logits = self.ddos_cls(lm_cls_h)
        vuln_logits = self.vuln_cls(lm_cls_h)
        uninfo_logits = self.uninfo_cls(lm_cls_h)
        nonsec_logits = self.nonsec_cls(lm_cls_h)

        return nonsec_logits, uninfo_logits, dp_logits, fp_logits, rm_logits, ddos_logits, vuln_logits
