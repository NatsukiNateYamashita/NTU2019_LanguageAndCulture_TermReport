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

# def pkl_dump(obj, f_name):
# def pkl_load(f_name):
# GPUが使えれば利用する設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###################################################### SET params
PRETRAINED_MODEL_NAME = "bert-base-chinese" # 12-layer, 768-hidden, 12-heads, 110M parameters. Trained on cased Chinese Simplified and Traditional text.
MAX_LENGTH = 352
NUM_LABEL = 4
BATCH_SIZE = 16
NUM_DIALOGUE = 4142
NUM_SENTENCES = 25548
MAX_EPOCH = 50
USE_TAG = "rel_posi"

###################################################### SET model
model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME,  
        num_labels=NUM_LABEL,  # ラベル数（今回はBinayなので2、数値を増やせばマルチラベルも対応可）
        output_attentions=True,  # アテンションベクトルを出力するか
        output_hidden_states=False,  # 隠れ層を出力するか
        )
# optimizer = AdamW(model.parameters(), lr=5e-6)
optimizer = AdamW(model.parameters(), lr=1e-6)
model.cuda()

###################################################### LOAD model
# output_model_dir = '../model_save/'
# model_PATH ='{}rel_posi_b_16_e_34'.format(output_model_dir)
# model_state_dict = torch.load(model_PATH)
# model = BertForSequenceClassification.from_pretrained(
#         PRETRAINED_MODEL_NAME,  
#         num_labels=NUM_LABEL,  # ラベル数（今回はBinayなので2、数値を増やせばマルチラベルも対応可）
#         output_attentions=True,  # アテンションベクトルを出力するか
#         output_hidden_states=False,  # 隠れ層を出力するか
#         state_dict=model_state_dict
#         )
# ###################################################### LOAD data
relation_list = pkl_load("relation_list")
# relations = pkl_load("relations")
# utterances = pkl_load("utterances")
# num_relation = len(relation_list)
# print(len(relation_list))

###################################################### COUNT max length of utterances
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
# # tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
# maxlen = []
# for u in utterances:
#         token_words = tokenizer.tokenize(str(u))
#         pprint(token_words)
#         maxlen.append(len(token_words))
# max_length = max(maxlen)
# print(max_length)
# # max_length: 350

# ###################################################### TOKENIZE
# input_ids = []
# attention_masks = []
# for utterance in utterances:
#         encoded_dict = tokenizer.encode_plus(
#                 utterance,
#                 add_special_tokens=True,  # Special Tokenの追加
#                 max_length=MAX_LENGTH,           # 文章の長さを固定（Padding/Trancatinating）
#                 pad_to_max_length=True,  # PADDINGで埋める
#                 return_attention_mask=True,   # Attention maksの作成
#                 return_tensors='pt',  # Pytorch tensorsで返す
#         )
#         input_ids.append(encoded_dict['input_ids'])
#         attention_masks.append(encoded_dict['attention_mask'])
# input_ids = torch.cat(input_ids, dim=0)
# attention_masks = torch.cat(attention_masks, dim=0)

# ###################################################### CREATE labels
rel_to_label_dict, label_to_rel_dict = create_mapping(relation_list)
# labels = []
# for rel in relations:
#     labels.append(rel_to_label_dict[rel])
# labels = torch.tensor(labels)

###################################################### CREATE labels_in_position
# ---------------------------------------------------
position_list = pkl_load("position_list")

superior = []
for rel in position_list["superior"]:
    superior.append(rel_to_label_dict[rel])
peer = []
for rel in position_list["peer"]:
    peer.append(rel_to_label_dict[rel])
inferior = []
for rel in position_list["inferior"]:
    inferior.append(rel_to_label_dict[rel])

# new_labels = []
# for l in labels:
#     if l in superior:
#         new_labels.append(0)
#     elif l in peer:
#         new_labels.append(1)
#     elif l in inferior:
#         new_labels.append(2)
#     else:
#         new_labels.append(3)

# labels = new_labels
# labels = torch.tensor(labels)
# ---------------------------------------------------

###################################################### DUMP data
# pkl_dump(input_ids, "posi_input_ids")
# pkl_dump(attention_masks, "posi_attention_masks")
# pkl_dump(labels, "labels")
# ---------------------------------------------------
# pkl_dump(labels, "posi_labels")
# ---------------------------------------------------

##################################################### LOAD data
# input_ids = pkl_load("input_ids")
# attention_masks = pkl_load("attention_masks")
# labels = pkl_load("labels")
# ---------------------------------------------------
labels = pkl_load("posi_labels")
# ---------------------------------------------------

# ###################################################### CREATE dataset
# dataset = TensorDataset(input_ids, attention_masks, labels)
# # dataset = TensorDataset(input_ids[:160], attention_masks[:160], labels[:160])
# # 90%:train 10%:validation
# train_size = int(0.8 * len(dataset))
# validation_size = len(dataset) - train_size
# print('train_data_set_size: {}'.format(train_size))
# print('validation_data_set_size:　{} '.format(validation_size))
# train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

# # train_dataloader
# train_dataloader = DataLoader(
# train_dataset,
# sampler=RandomSampler(train_dataset),  # ランダムにデータを取得してバッチ化
# batch_size=BATCH_SIZE
# )
# # vilidation_dataloader
# validation_dataloader = DataLoader(
# validation_dataset,
# sampler=SequentialSampler(validation_dataset),  # 順番にデータを取得してバッチ化
# batch_size = BATCH_SIZE
# )
# ###################################################### DUMP data
# pkl_dump(train_dataset, "train_dataset")
# pkl_dump(validation_dataset, "validation_dataset")
# pkl_dump(train_dataloader, "train_dataloader")
# pkl_dump(validation_dataloader, "validation_dataloader")
# ---------------------------------------------------
# pkl_dump(train_dataset, "posi_train_dataset")
# pkl_dump(validation_dataset, "posi_validation_dataset")
# pkl_dump(train_dataloader, "posi_train_dataloader")
# pkl_dump(validation_dataloader, "posi_validation_dataloader")
# ---------------------------------------------------

##################################################### LOAD data
# ---------------------------------------------------
# train_dataset = pkl_load("train_dataset")
# validation_dataset = pkl_load("validation_dataset")
# train_dataloader = pkl_load("train_dataloader")
# validation_dataloader = pkl_load("validation_dataloader")
# ---------------------------------------------------
train_dataset = pkl_load("posi_train_dataset")
validation_dataset = pkl_load("posi_validation_dataset")
train_dataloader = pkl_load("posi_train_dataloader")
validation_dataloader = pkl_load("posi_validation_dataloader")

###################################################### DEFINE train function
def train(model):
        model.train()  
        train_loss = 0
        for batch in train_dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                optimizer.zero_grad()
                loss, logits, attentions = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
        return train_loss

###################################################### DEFINE validation function
def validation(model,batch_size,epoch):
        model.eval()
        df = pd.DataFrame()
        attentions_list = []
        with torch.no_grad():
            for i, batch in enumerate(validation_dataloader):
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                preds, attentioons = model( b_input_ids,
                                            token_type_ids=None)
                pred_df = pd.DataFrame(np.argmax(preds.cpu().numpy(), axis=1))
                label_df = pd.DataFrame(b_labels.cpu().numpy())
                answer_df = pd.concat([pred_df, label_df], axis=1)
                df = pd.concat([df,answer_df])
        # CULCURATE accuracy
        df.columns=['pred_label','true_label']
        accuracy = accuracy_score(df['true_label'],df['pred_label'])

        # Convert to posi
        df.replace(0, 'superior')
        df.replace(1, 'peer')
        df.replace(2, 'inferior')
        df.replace(3, 'unknown')

        # SAVE predict labels and labels
        output_prediction_dir = '../csv_save/'
        prediction_PATH ='{}{}_b_{}_e_{}.csv'.format(output_prediction_dir, USE_TAG, BATCH_SIZE, epoch)
        df.to_csv(prediction_PATH, sep=",",index=False)   
        
        # SAVE model
        output_model_dir = '../model_save/'
        model_PATH ='{}{}_b_{}_e_{}'.format(output_model_dir, USE_TAG, BATCH_SIZE, epoch)
        torch.save(model.state_dict(), model_PATH)  

        return accuracy

###################################################### FINETUNE
log = []
log.append(['epoch', 'loss', 'accuracy'])
for epoch in range(MAX_EPOCH):
        accuracy = validation(model,BATCH_SIZE,epoch)
        loss = train(model)
        log.append([epoch, loss, accuracy])
        print("epoch: {}, loss: {}, accuracy: {}".format(epoch,loss,accuracy))

# SAVE log
log_df = pd.DataFrame(log)
output_log_dir = '../log_save/'
log_PATH ='{}{}_b_{}.csv'.format(output_log_dir, USE_TAG, BATCH_SIZE)
log_df.to_csv(log_PATH, sep=",", index=False)

# SAVE graph
sns.set()
sns.set_style("whitegrid", {'grid.linestyle': '--'})
sns.set_context("paper", 1.5, {"lines.linewidth": 4})
sns.set_palette("winter_r", 8, 1)
sns.set('talk', 'whitegrid', 'dark', font_scale=1.5,
        rc={"lines.linewidth": 2, 'grid.linestyle': '--'})

plt.plot(log[1:,0],log[1:,2])
png_PATH ='{}{}_b_{}.png'.format(output_log_dir, USE_TAG, BATCH_SIZE)
plt.savefig(png_PATH)