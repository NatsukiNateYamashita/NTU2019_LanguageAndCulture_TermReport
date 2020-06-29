from myutil import *
from pprint import pprint
import csv


# USE_TAG = "rel"
USE_TAG = "rel_posi"
# VALIDATION_DATASET = "validation_dataset"
VALIDATION_DATASET = "posi_validation_dataset"
# BATCH_SIZE_FOR_GET_ATTENTION = 1
# EPOCH = 7
EPOCH = 2
BATCH_SIZE = 16
POSITION = "superior"

# output_correct_dir = '../correct_set/'
# correct_PATH ='{}correct_{}_b_{}_e_{}.csv'.format(output_correct_dir, USE_TAG, BATCH_SIZE, EPOCH)      
# with open(correct_PATH, 'r') as f:
#     reader = csv.reader(f)
#     correct_ = [row for row in reader]
# pprint(correct_)

# output_error_dir = '../error_set/'
# error_PATH ='{}error_{}_b_{}_e_{}.csv'.format(output_error_dir, USE_TAG, BATCH_SIZE, EPOCH)      
# with open(error_PATH, 'r') as f:
#     reader = csv.reader(f)
#     error_ = [row for row in reader]
# pprint(error_)

# output_correct_dir = '../correct_set/'
# correct_PATH ='{}correct_{}_{}_b_{}_e_{}.csv'.format(output_correct_dir, POSITION, USE_TAG, BATCH_SIZE, EPOCH)      
# with open(correct_PATH, 'r') as f:
#     reader = csv.reader(f)
#     correct_type = [row for row in reader]
# print(correct_type)

output_error_dir = '../error_set/'
error_PATH ='{}error_{}_{}_b_{}_e_{}.csv'.format(output_error_dir, POSITION, USE_TAG, BATCH_SIZE, EPOCH)      
with open(error_PATH, 'r') as f:
    reader = csv.reader(f)
    error_type = [row for row in reader]
pprint(error_type)
