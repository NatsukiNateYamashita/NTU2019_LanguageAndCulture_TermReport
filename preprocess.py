from myutil import *
from collections import OrderedDict
from pprint import pprint
import json
import pandas as pd
from sklearn.metrics import classification_report
# def pkl_dump(obj, f_name):
# def pkl_load(f_name):
###################################################### LOAD metadata
f_path = "../mpdd/metadata.json"
with open(f_path, 'r') as f:
    metadata = json.load(f)
# pprint(metadata)
relation_list = metadata['relation']
field_list = metadata['field']
position_list = metadata['position']
# pprint(relation_list)
# pprint(position_list)
###################################################### LOAD dialogue
f_path = "../mpdd/dialogue.json"
with open(f_path, 'r') as f:
    dialogue = json.load(f)
relations = []
utterances = []
for_print = []
for key, value in dialogue.items():
    for val in value:
        # print(val)
        # pprint(val['listener'][0]['relation'])
        # pprint(val['utterance'])
        # print()
        # if val['listener'][0]['relation'] not in relation_list:
        #     print(val['listener'][0]['relation'])
        relations.append(val['listener'][0]['relation'])
        utterances.append(val['utterance'])
        if val['listener'][0]['relation'] == "child":
            for_print.append(val['utterance']+"\n")
with open("child_utterance.csv",'w',encoding='utf-8-sig') as f:
    ba = 0
    ma = 0
    zhengpeng = 0

    for p in for_print:
        if "！" in p:
            ba += 1
        if "？" in p:
            ma += 1
        # if "正鵬" in p:
        #     zhengpeng += 1
        f.writelines(p)
print("ba: {}, ma: {}".format(ba,ma))

# # of dialogue: 4142
# # of sentence: 25548
# pkl_dump(relation_list, "relation_list")
# pkl_dump(field_list, "field_list")
# pkl_dump(position_list, "position_list")
# pkl_dump(relations, "relations")
# pkl_dump(utterances, "utterances")

# s_c = 0
# p_c = 0
# i_c = 0
# for rel in relations:
#     if rel in position_list["superior"]:
#         s_c+=1
#     if rel in position_list["peer"]:
#         p_c+=1
#     if rel in position_list["inferior"]:
#         i_c+=1
# print("superior: ", s_c)
# print("peer:     ", p_c)
# print("inferior: ", i_c)

###Search word in utterances
# s_c=0
# other_c=0
# search_char = "睡"
# search_char_2 = "醒"
# s_u = []
# other_u = []
# for i, u in enumerate(utterances):
#     if (search_char in u) & (search_char_2 in u):
#         if relations[i] in position_list["superior"]:
#             s_u.append(u)
#             s_c+=1
#         else:
#             other_c+=1
#             other_u.append(u)
# print("{}, {}in superior: {}".format(search_char,search_char_2,s_c))
# print("{}, {}in other   : {}".format(search_char,search_char_2,other_c))
# # pprint("{}".format(s_u))
# # pprint("\n other_u   : {}".format(other_u))

# s_u = []
# for i,rel in enumerate(relations):
#     if rel in position_list["superior"]:
#         s_u.append(utterances[i])

# pprint(s_u)

# df = pd.DataFrame(relations)
# df = classification_report(df,df)
# with open("for_report.csv",'w') as f:
#     for line in df:
#         f.writelines(line)