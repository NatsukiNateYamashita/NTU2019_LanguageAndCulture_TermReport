from myutil import *
from pprint import pprint
import transformers
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig,AutoTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import csv
from sklearn.metrics import classification_report

###################################################### SET params
USE_TAG = "rel"
# USE_TAG = "rel_posi"
VALIDATION_DATASET = "validation_dataset"
# VALIDATION_DATASET = "posi_validation_dataset"
# BATCH_SIZE_FOR_GET_ATTENTION = 1
EPOCH = 7
# EPOCH = 2
BATCH_SIZE = 16


# ###################################################### MAKE mapping list
relation_list = pkl_load("relation_list")
rel_to_label_dict, label_to_rel_dict = create_mapping(relation_list)

# ###################################################### Make pred true list
csv_path = "../csv_save/{}_b_{}_e_{}.csv".format(USE_TAG,BATCH_SIZE,EPOCH)
with open(csv_path, "r") as f:
    reader = csv.reader(f)
    pred_true_label = [row for row in reader]
pred_true_label = pred_true_label[1:]
print(len(pred_true_label))
###############################################################################################
###############################################################################################
###############################################################################################
if USE_TAG == "rel":
    # print(pred_true_label)
    # ###################################################### Convert label to rel
    pred_true =[]
    for p, t in pred_true_label:
        pred = label_to_rel_dict[int(p)]
        true = label_to_rel_dict[int(t)]
        pred_true.append([pred, true])

    # ###################################################### SAVE correct_set and error_set
    
    utterances = pkl_load("utterances")
    correct_set = []
    error_set = []
    for i, [p, t] in enumerate(pred_true):
        pkl_f_name = "utterance/{}_b_{}_e_{}_i_{}".format(USE_TAG, BATCH_SIZE, EPOCH ,i)
        pkl_path = "../pkl/{}".format(pkl_f_name)
        with open(pkl_path, 'rb') as f:
            utterances = pickle.load(f)
        if p == t:
            correct_set.append([i,p,t,utterances])
        else:
            error_set.append([i,p,t,utterances])

    output_correct_dir = '../correct_set/'
    correct_PATH ='{}correct_{}_b_{}_e_{}.csv'.format(output_correct_dir, USE_TAG, BATCH_SIZE, EPOCH)      
    with open(correct_PATH, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(correct_set)

    output_error_dir = '../error_set/'
    error_PATH ='{}error_{}_b_{}_e_{}.csv'.format(output_error_dir, USE_TAG, BATCH_SIZE, EPOCH)      
    with open(error_PATH, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(error_set)

    make_detail_csv(correct_set, error_set,USE_TAG, BATCH_SIZE, EPOCH)

    # ###################################################### PRINT set
    # print("---------- CORRECT -----------")
    # for i, [p,t] in enumerate(correct_set):
    #     pprint("\npred: {}, \ntrue: {}, \nutterance: {}".format(p,t,utterances[i]))

    # print("---------- ERROR -----------")
    # for i, [p,t] in enumerate(error_set):
    #     print("\npred: {}, \ntrue: {}, \nutterance: {}".format(p,t,utterances[i]))

    # ###################################################### SAVE classification result
    df = pd.DataFrame(pred_true, columns=['pred','true'])
    df =classification_report(df['pred'], df['true'])
    print(df)
    output_report_dir = '../report/'
    report_PATH ='{}classification_report_{}_b_{}_e_{}.csv'.format(output_report_dir, USE_TAG, BATCH_SIZE, EPOCH)
    df.to_csv(report_PATH)

    # ###################################################### SAVE classification result
    # ###################################################### SAVE classification result

    position_list = pkl_load("position_list")
    print(pred_true[:10])
    pred_true_converted = []
    for p,t in pred_true:
        if p in position_list["superior"]:
            p = "superior"
        elif p in position_list["peer"]:
            p = "peer"
        elif p in position_list["inferior"]:
            p = "inferior"
        else:
            p = "unknown"
        if t in position_list["superior"]:
            t = "superior"
        elif t in position_list["peer"]:
            t = "peer"
        elif t in position_list["inferior"]:
            t = "inferior"
        else:
            t = "unknown"
        pred_true_converted.append([p,t])

    # print(pred_true_converted)

    df = pd.DataFrame(pred_true_converted,columns=['pred','true'])
    df =classification_report(df['pred'], df['true'])
    print(df)
    output_report_dir = '../report/'
    report_PATH ='{}classification_report_rel2posi_{}_b_{}_e_{}.csv'.format(output_report_dir, USE_TAG, BATCH_SIZE, EPOCH)
    df.to_csv(report_PATH)



###############################################################################################
###############################################################################################
###############################################################################################

elif USE_TAG == "rel_posi":
    # ###################################################### Convert label to rel
    label_to_rel_dict = {0:'superior',1:'peer',2:'inferior',3:'unknown'}
    pred_true =[]
    for p, t in pred_true_label:
        pred = label_to_rel_dict[int(p)]
        true = label_to_rel_dict[int(t)]
        pred_true.append([pred, true])
    print(len(pred_true))
    # ###################################################### SAVE correct_set and error_set
    # utterances = pkl_load("utterances")
    correct_set = []
    error_set = []
    for i, [p, t] in enumerate(pred_true):
        pkl_f_name = "utterance/{}_b_{}_e_{}_i_{}".format(USE_TAG, BATCH_SIZE, EPOCH ,i)
        pkl_path = "../pkl/{}".format(pkl_f_name)
        with open(pkl_path, 'rb') as f:
            utterances = pickle.load(f)
        if p == t:
            correct_set.append([i,p,t,utterances])
        else:
            error_set.append([i,p,t,utterances])

    output_correct_dir = '../correct_set/'
    correct_PATH ='{}correct_{}_b_{}_e_{}.csv'.format(output_correct_dir, USE_TAG, BATCH_SIZE, EPOCH)      
    with open(correct_PATH, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(correct_set)

    output_error_dir = '../error_set/'
    error_PATH ='{}error_{}_b_{}_e_{}.csv'.format(output_error_dir, USE_TAG, BATCH_SIZE, EPOCH)      
    with open(error_PATH, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(error_set)

    make_detail_csv(correct_set, error_set,USE_TAG, BATCH_SIZE, EPOCH)
    # ###################################################### SAVE classification result
    df = pd.DataFrame(pred_true,columns=['pred','true'])
    df =classification_report(df['pred'], df['true'])
    print(df)
    output_report_dir = '../report/'
    report_PATH ='{}classification_report_{}_b_{}_e_{}.csv'.format(output_report_dir, USE_TAG, BATCH_SIZE, EPOCH)
    df.to_csv(report_PATH)