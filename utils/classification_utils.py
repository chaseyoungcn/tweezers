import os 
import pandas as pd
import numpy as np
from pandarallel import pandarallel
import pickle
from tqdm import tqdm
tqdm.pandas()

import torch
from torch.utils.data import TensorDataset, DataLoader, \
RandomSampler, SequentialSampler

from transformers import AutoTokenizer

from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc 

from skmultilearn.model_selection import iterative_train_test_split

from model.tweetclassifier import TweetClassifier
from utils.preprocess_text import preprocess_text

from pdb import set_trace

EVENT_TYPES = [    
    'Not Security related', 'Uninformative', 'Data Privacy',
    'Fraud/Phishing', 'Ransomware/Malware', 'DDoS', 'Vulnerability'
]
ET2IDX = {
    'Not Security related': 0, 
    'Uninformative':1, 
    'Data Privacy':2,
    'Fraud/Phishing':3, 
    'Ransomware/Malware':4, 
    'DDoS':5, 
    'Vulnerability':6 
}

def get_model_eval_result(df, rs, lm, device, batch_size=64):
    # Shuffle dataset
    df = df.sample(frac=1, random_state=rs)

    # Make X, Y 
    X = df['text_p'].to_numpy()
    X_idx = np.array([[i,0] for i in range(X.shape[0])])
    y = df[EVENT_TYPES].values
        
    # Split train, val, test
    idx_fname = f'data/idx_split_{rs}.pickle'
    if os.path.exists(idx_fname):
        with open(idx_fname, 'rb') as f:
            X_idx_train, y_train,  X_idx_val, y_val, X_idx_test, y_test = pickle.load(f) 
    else:
        X_idx_train, y_train, X_idx_test, y_test = iterative_train_test_split(X_idx, y, test_size=0.2)
        X_idx_train, y_train, X_idx_val, y_val = iterative_train_test_split(X_idx_train, y_train, test_size=0.25)
        with open(idx_fname, 'wb') as f:
            pickle.dump([X_idx_train, y_train,  X_idx_val, y_val, X_idx_test, y_test], f)

    # Init tokenizer
    if lm == "bert-base-cased":
        print('load bert base cased tokenizer')
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    elif lm == "bert-base-uncased":
        print('load bert base uncased tokenizer')
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    elif lm == 'roberta-base':
        print('load roberta base tokenizer')
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    elif lm == 'bertweet':
        print('load bertweet tokenizer')
        tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
    elif lm == 'securebert':
        print('load secubert tokenizer')
        tokenizer = AutoTokenizer.from_pretrained('ehsanaghaei/SecureBERT')     
    else:
        print("SHOULD NOT HAPPEN")
        
    # Specify loss function
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8]).to(device)) 
    model_path = f'trained_models/tweet_multi_cls_{lm}_{rs}.pt'

    model = TweetClassifier(lm)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Tokenize the text to make model inputs
    test_inputs = tokenizer(
        X[X_idx_test[:,0]].tolist(), return_tensors="pt",
        padding=True, truncation=True)
    test_labels = torch.tensor(y_test)

    # # Create the DataLoader for our validation set
    test_data = TensorDataset(
        torch.tensor(X_idx_test[:,0]),
        test_inputs['input_ids'],
        test_inputs['attention_mask'],
        test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(
        test_data,
        sampler=test_sampler,
        batch_size=batch_size)

    logits_list, labels_list, input_idx_list, val_loss = evaluate(model, test_dataloader, loss_fn, device)

    logits_list = [logits.cpu() for logits in logits_list]
    labels_list = [labels.cpu() for labels in labels_list]
    preds_list = [torch.argmax(logits, dim=1).flatten() for logits in logits_list]

    return input_idx_list, logits_list, labels_list, preds_list

def process_raw_df(
    lm, rs, 
    df, cuda,
    ):
    device = torch.device('cpu') if cuda==-1 \
        else torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

    print(f"classfier is on {device}")
    model_path = f'../trained_models/tweet_multi_cls_{lm}_{rs}.pt'

    model = TweetClassifier(lm)
    model.load_state_dict(torch.load(model_path,  map_location=device), strict=False)
    model.to(device)
    
    # Init tokenizer
    if lm == "bert-base-cased":
        print('load bert base cased tokenizer')
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    elif lm == "bert-base-uncased":
        print('load bert base uncased tokenizer')
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    elif lm == 'roberta-base':
        print('load roberta base tokenizer')
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    elif lm == 'bertweet':
        print('load bertweet tokenizer')
        tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
    else:
        print("SHOULD NOT HAPPEN")
        
    X = df['text_p'].to_numpy()
    y = torch.tensor(([[0]*7 for _ in range(df.shape[0])])) # dummy y 
    

    inputs = tokenizer(
        X.tolist(), return_tensors="pt",
        padding=True, truncation=True)
    test_data = TensorDataset(
        y,
        inputs['input_ids'],
        inputs['attention_mask'],
        y,
    )
    test_sampler = SequentialSampler(test_data)

    test_dataloader = DataLoader(
        test_data,
        sampler=test_sampler,
        batch_size=4096)

    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8]).to(device))         

    print("Computing event type...")
    logits_list, labels_list, input_idx_list, val_loss = evaluate(model, test_dataloader, loss_fn, device)

    logits_list = [logits.cpu() for logits in logits_list]
    labels_list = [labels.cpu() for labels in labels_list]
    preds_list = [torch.argmax(logits, dim=1).flatten() for logits in logits_list]
    preds_list = [preds.unsqueeze(dim=1) for preds in preds_list]
    preds_matrix = torch.cat(preds_list, dim=1)

    for i, et in enumerate(EVENT_TYPES):
        df[et] = preds_matrix[:, i]

    return df


def get_pred_result_from_text(test_text, model, tokenizer, cuda=0):

    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    test_inputs = tokenizer(
        test_text, return_tensors="pt",
        padding=True, truncation=True).to(device)
    model.eval()
    with torch.no_grad():
        logits_list = model(**test_inputs)
    # print(logits_list)
    preds_list = [int(torch.argmax(logit, dim=1).flatten().cpu()) for logit in logits_list]
    preds_list_ = [float(torch.nn.functional.softmax(logit).cpu()[:,1]) for logit in logits_list]

    return preds_list, preds_list_, [et for et, p in zip(EVENT_TYPES, preds_list) if p==1]


def load_classification_dataset(file_path='data/tweet_mix_r1r2_agg.xlsx'):
    df = pd.read_excel(file_path)
    df = df.dropna(how='all')
    df = df.fillna(0)
    df = df.replace('o', 1)
    df = df.replace('O', 1)
    df = df.replace('?', 1)

    return df

def evaluate(
    model, 
    dataloader,
    loss_fn,
    device):

    model.eval()

    all_dp_logits, all_fp_logits, all_rm_logits, \
        all_ddos_logits, all_vuln_logits, all_uninfo_logits, all_nonsec_logits  = [], [], [], [], [], [], []
    all_dp_labels, all_fp_labels, all_rm_labels, \
        all_ddos_labels, all_vuln_labels, all_uninfo_labels, all_nonsec_labels  = [], [], [], [], [], [], []
    input_idx_list = []

    for i, batch in enumerate(tqdm(dataloader)):
        # Load batch to GPU
        b_input_idx, b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        input_idx_list.append(b_input_idx)

        with torch.no_grad():
            nonsec_logits, uninfo_logits, dp_logits, fp_logits, rm_logits, ddos_logits, vuln_logits = model(b_input_ids, b_attn_mask)

        all_nonsec_logits.append(nonsec_logits)
        all_uninfo_logits.append(uninfo_logits)
        all_dp_logits.append(dp_logits)
        all_fp_logits.append(fp_logits)
        all_rm_logits.append(rm_logits)
        all_ddos_logits.append(ddos_logits)
        all_vuln_logits.append(vuln_logits)

        all_nonsec_labels.append(b_labels[:,ET2IDX['Not Security related']])
        all_uninfo_labels.append(b_labels[:,ET2IDX['Uninformative']])
        all_dp_labels.append(b_labels[:,ET2IDX['Data Privacy']])
        all_fp_labels.append(b_labels[:,ET2IDX['Fraud/Phishing']])
        all_rm_labels.append(b_labels[:,ET2IDX['Ransomware/Malware']])
        all_ddos_labels.append(b_labels[:,ET2IDX['DDoS']])
        all_vuln_labels.append(b_labels[:,ET2IDX['Vulnerability']])

    input_idx_list = torch.cat(input_idx_list)
    
    # Concatenate logits from each batch
    all_nonsec_logits = torch.cat(all_nonsec_logits, dim=0)
    all_nonsec_preds = torch.argmax(all_nonsec_logits, dim=1).flatten()
    all_nonsec_labels = torch.cat(all_nonsec_labels, dim=0)
    
    all_uninfo_logits = torch.cat(all_uninfo_logits, dim=0)
    all_uninfo_preds = torch.argmax(all_uninfo_logits, dim=1).flatten()
    all_uninfo_labels = torch.cat(all_uninfo_labels, dim=0)

    all_dp_logits = torch.cat(all_dp_logits, dim=0)
    all_dp_preds = torch.argmax(all_dp_logits, dim=1).flatten()
    all_dp_labels = torch.cat(all_dp_labels, dim=0)

    all_fp_logits = torch.cat(all_fp_logits, dim=0)
    all_fp_preds = torch.argmax(all_fp_logits, dim=1).flatten()
    all_fp_labels = torch.cat(all_fp_labels, dim=0)

    all_rm_logits = torch.cat(all_rm_logits, dim=0)
    all_rm_preds = torch.argmax(all_rm_logits, dim=1).flatten()
    all_rm_labels = torch.cat(all_rm_labels, dim=0)

    all_ddos_logits = torch.cat(all_ddos_logits, dim=0)
    all_ddos_preds = torch.argmax(all_ddos_logits, dim=1).flatten()
    all_ddos_labels = torch.cat(all_ddos_labels, dim=0)

    all_vuln_logits = torch.cat(all_vuln_logits, dim=0)
    all_vuln_preds = torch.argmax(all_vuln_logits, dim=1).flatten()
    all_vuln_labels = torch.cat(all_vuln_labels, dim=0)

    # Compute validation loss
    val_loss = loss_fn(all_nonsec_logits, all_nonsec_labels.to(device))
    val_loss += loss_fn(all_uninfo_logits, all_uninfo_labels.to(device))
    val_loss += loss_fn(all_dp_logits, all_dp_labels.to(device))
    val_loss += loss_fn(all_fp_logits, all_fp_labels.to(device))
    val_loss += loss_fn(all_rm_logits, all_rm_labels.to(device))
    val_loss += loss_fn(all_ddos_logits, all_ddos_labels.to(device))
    val_loss += loss_fn(all_vuln_logits, all_vuln_labels.to(device))

    return [all_nonsec_logits, all_uninfo_logits, all_dp_logits, all_fp_logits, all_rm_logits, all_ddos_logits, all_vuln_logits], \
        [all_nonsec_labels, all_uninfo_labels, all_dp_labels, all_fp_labels, all_rm_labels, all_ddos_labels, all_vuln_labels], input_idx_list, val_loss

# def get_classification_results(all_preds, all_logits, all_labels, draw=False):
    
#     fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_logits[:,1])
#     roc_auc = metrics.auc(fpr, tpr)
#     # print(roc_auc, roc_auc_score(all_labels, all_logits[:,1]))
#     if draw:
#         draw_curve(fpr, tpr, roc_auc)
    
#     precisions, recalls, thresholds = precision_recall_curve(all_labels, all_logits[:,1])

#     pr_auc = auc(recalls, precisions)
#     if draw:
#         draw_curve(recalls, precisions, pr_auc)

#     precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
#     # f1 = f1_score(all_labels, all_preds)
#     acc = accuracy_score(all_labels, all_preds)


#     return precision, recall, f1, acc, roc_auc, pr_auc
