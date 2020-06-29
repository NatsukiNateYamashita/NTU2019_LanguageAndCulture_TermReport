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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_mapping(item):
	"""
	Create a mapping (item to ID / ID to item) from a dictionary.
	Items are ordered by decreasing frequency.
	"""
	id_to_item = {i: v for i, v in enumerate(item)}
	item_to_id = {v: k for k, v in id_to_item.items()}
	return item_to_id, id_to_item

###################################################### SET params
PRETRAINED_MODEL_NAME = "bert-base-chinese" # 12-layer, 768-hidden, 12-heads, 110M parameters. Trained on cased Chinese Simplified and Traditional text.
MAX_LENGTH = 352
# NUM_LABEL = 25
NUM_LABEL = 4
BATCH_SIZE = 16
NUM_DIALOGUE = 4142
NUM_SENTENCES = 25548
MAX_EPOCH = 50

###################################################### SET params
# USE_TAG = "rel"
USE_TAG = "rel_posi"
# VALIDATION_DATASET = "validation_dataset"
VALIDATION_DATASET = "posi_validation_dataset"
BATCH_SIZE_FOR_GET_ATTENTION = 1
# epoch = 7
epoch = 2

###################################################### SET model
output_model_dir = '../model_save/'
model_PATH ='{}{}_b_{}_e_{}'.format(output_model_dir, USE_TAG, BATCH_SIZE, epoch)
model_state_dict = torch.load(model_PATH)
model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME,  
        num_labels=NUM_LABEL,  # ラベル数（今回はBinayなので2、数値を増やせばマルチラベルも対応可）
        output_attentions=True,  # アテンションベクトルを出力するか
        output_hidden_states=False,  # 隠れ層を出力するか
        state_dict=model_state_dict
        )
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
optimizer = AdamW(model.parameters(), lr=1e-6)
model.cuda()

# ###################################################### CREATE converting dict
relation_list = pkl_load("relation_list")
rel_to_label_dict, label_to_rel_dict = create_mapping(relation_list)

# ###################################################### LOAD validation data
validation_dataset = pkl_load(VALIDATION_DATASET)
validation_dataloader = DataLoader(
validation_dataset,
sampler=SequentialSampler(validation_dataset),  # 順番にデータを取得してバッチ化
batch_size = BATCH_SIZE_FOR_GET_ATTENTION
)
# ###################################################### DEFINE validation
def validation(model,batch_size,epoch):
        model.eval()
        df = pd.DataFrame()
        attentions_list = []
        with torch.no_grad():
            for i, batch in enumerate(validation_dataloader):
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                preds, attentions = model( b_input_ids,
                                            token_type_ids=None)
                
                # GET attentions and CONVERT to list
                pred_df = pd.DataFrame(np.argmax(preds.cpu().numpy(), axis=1))
                label_df = pd.DataFrame(b_labels.cpu().numpy())
                answer_df = pd.concat([pred_df, label_df], axis=1)
                df = pd.concat([df,answer_df])
                temp = []
                for a in attentions:
                    a = a.cpu()
                    temp.append(a)
                attentions = temp

                # SAVE utterance  
                input_id_list = b_input_ids[0].tolist() # Batch index 0
                utterance = tokenizer.convert_ids_to_tokens(input_id_list)
                output_utterance_dir ='utterance/'
                utterance_PATH = '{}{}_b_{}_e_{}_i_{}'.format(output_utterance_dir, USE_TAG, BATCH_SIZE, epoch, i)
                pkl_dump(utterance, utterance_PATH)  

                # SAVE attention
                output_attention_dir ='attention/'
                attention_PATH ='{}{}_b_{}_e_{}_i_{}'.format(output_attention_dir, USE_TAG, BATCH_SIZE, epoch, i)
                pkl_dump(attentions, attention_PATH) 

                # CULCURATE accuracy
                pred_df = pd.DataFrame(np.argmax(preds.cpu().numpy(), axis=1))
                label_df = pd.DataFrame(b_labels.cpu().numpy())
                answer_df = pd.concat([pred_df, label_df], axis=1)
                df = pd.concat([df,answer_df])

        # CULCURATE accuracy
        df.columns=['pred_label','true_label']
        accuracy = accuracy_score(df['true_label'],df['pred_label'])

        # label_to_rel_
        if USE_TAG == "rel":      
            df.replace(label_to_rel_dict)
        # label_to_rel
        elif USE_TAG == "rel_posi": 
            df.replace(0, 'superior')
            df.replace(1, 'peer')
            df.replace(2, 'inferior')
            df.replace(3, 'unknown')

        # SAVE predict labels and labels
        output_prediction_dir = '../csv_save/'
        prediction_PATH ='{}{}_b_{}_e_{}.csv'.format(output_prediction_dir, USE_TAG, BATCH_SIZE, epoch)
        df.to_csv(prediction_PATH, sep=",",index=False)   

        return accuracy

# ###################################################### RUN
accuracy = validation(model,BATCH_SIZE,epoch)