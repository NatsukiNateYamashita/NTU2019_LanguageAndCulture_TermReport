from myutil import *
from pprint import pprint
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig,AutoTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import csv

def create_mapping(item):
	"""
	Create a mapping (item to ID / ID to item) from a dictionary.
	Items are ordered by decreasing frequency.
	"""
	id_to_item = {i: v for i, v in enumerate(item)}
	item_to_id = {v: k for k, v in id_to_item.items()}
	return item_to_id, id_to_item

# ###################################################### LOAD data
relation_list = pkl_load("relation_list")
rel_to_label_dict, label_to_rel_dict = create_mapping(relation_list)

utterances = pkl_load("utterances")

csv_path = "../csv_save/rel_b_16_e_46.csv"
with open(csv_path, "r") as f:
    reader = csv.reader(f)
    pred_true_label = [row for row in reader]

pred_true_label = pred_true_label[1:]

# ###################################################### label to rel
pred_true =[]
for p, t in pred_true_label:
    pred = label_to_rel_dict[int(p)]
    true = label_to_rel_dict[int(t)]
    pred_true.append([pred, true])

# ###################################################### separate correct and error

correct_set = []
error_set = []
for i, [p, t] in enumerate(pred_true):
    if p == t:
        correct_set.append([i,p,t])
    else:
        error_set.append([i,p,t])

# ###################################################### output set
# for i,p,t in correct_set:
#     print("---------- CORRECT -----------")
#     pprint("pred: {}, true: {}, utterance: {}".format(p,t,utterances[i]))
    print("---------- ERROR -----------")
for i,p,t in error_set:
    print("\npred: {}, \ntrue: {}, \nutterance: {}".format(p,t,utterances[i]))
print("error: {} / {}".format(len(error_set),len(pred_true)))

