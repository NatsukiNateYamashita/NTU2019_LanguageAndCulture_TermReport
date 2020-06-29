
import pickle
import csv

####################################################### pickle 
def pkl_dump(obj, f_name):
    pkl_obj = obj
    pkl_f_name = f_name
    pkl_path = "../pkl/{}".format(pkl_f_name)
    with open(pkl_path, 'wb') as f:
        pickle.dump(pkl_obj , f)
    print("\nPicked {} !!!\n".format(pkl_path))

def pkl_load(f_name):
    pkl_f_name = str(f_name)
    pkl_path = "../pkl/{}".format(pkl_f_name)
    with open(pkl_path, 'rb') as f:
        pkl_obj = pickle.load(f)
    print("\nLoaded {} !!!\n".format(pkl_path))
    return pkl_obj

def create_mapping(item):
	"""
	Create a mapping (item to ID / ID to item) from a dictionary.
	Items are ordered by decreasing frequency.
	"""
	id_to_item = {i: v for i, v in enumerate(item)}
	item_to_id = {v: k for k, v in id_to_item.items()}
	return item_to_id, id_to_item

# hoge = "Hello!"
# pkl_dump(hoge, "hoge")

# hogehoge = pkl_load("hoge")
# print("hogehoge: {}".format(hogehoge))


def make_detail_csv(correct_,error_,USE_TAG, BATCH_SIZE, EPOCH):
    position_list = pkl_load("position_list")
    superior = []
    for rel in position_list["superior"]:
        superior.append(rel)
    peer = []
    for rel in position_list["peer"]:
        peer.append(rel)
    inferior = []
    for rel in position_list["inferior"]:
        inferior.append(rel)

    output_correct_dir = '../correct_set/'
    correct_PATH ='{}correct_{}_b_{}_e_{}.csv'.format(output_correct_dir, USE_TAG, BATCH_SIZE, EPOCH)      
    with open(correct_PATH, 'r') as f:
        reader = csv.reader(f)
        correct_ = [row for row in reader]

    output_error_dir = '../error_set/'
    error_PATH ='{}error_{}_b_{}_e_{}.csv'.format(output_error_dir, USE_TAG, BATCH_SIZE, EPOCH)      
    with open(error_PATH, 'r') as f:
        reader = csv.reader(f)
        error_ = [row for row in reader]

    # print(pred_true_label)
    correct_superior = []
    correct_peer = []
    correct_inferior = []
    correct_unknown = []
    for item in correct_:
        if item[1] in superior:
            correct_superior.append(item)
        elif item[1] in peer:
            correct_peer.append(item)
        elif item[1] in inferior:
            correct_inferior.append(item)
        elif item[1] == "superior":
            correct_superior.append(item)
        elif item[1] == "peer":
            correct_peer.append(item)
        elif item[1] == "inferior":
            correct_inferior.append(item)
        else:
            correct_unknown.append(item)

    output_correct_dir = '../correct_set/'
    correct_PATH ='{}correct_superior_{}_b_{}_e_{}.csv'.format(output_correct_dir, USE_TAG, BATCH_SIZE, EPOCH)      
    with open(correct_PATH, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(correct_superior)

    output_correct_dir = '../correct_set/'
    correct_PATH ='{}correct_peer_{}_b_{}_e_{}.csv'.format(output_correct_dir, USE_TAG, BATCH_SIZE, EPOCH)      
    with open(correct_PATH, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(correct_peer)

    output_correct_dir = '../correct_set/'
    correct_PATH ='{}correct_inferior_{}_b_{}_e_{}.csv'.format(output_correct_dir, USE_TAG, BATCH_SIZE, EPOCH)      
    with open(correct_PATH, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(correct_inferior)

    output_correct_dir = '../correct_set/'
    correct_PATH ='{}correct_unknown_{}_b_{}_e_{}.csv'.format(output_correct_dir, USE_TAG, BATCH_SIZE, EPOCH)      
    with open(correct_PATH, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(correct_unknown)

    error_superior = []
    error_peer = []
    error_inferior = []
    error_unknown = []
    for item in error_:
        if item[1] in superior:
            error_superior.append(item)
        elif item[1] in peer:
            error_peer.append(item)
        elif item[1] in inferior:
            error_inferior.append(item)
        elif item[1] == "superior":
            error_superior.append(item)
        elif item[1] == "peer":
            error_peer.append(item)
        elif item[1] == "inferior":
            error_inferior.append(item)
        else:
            error_unknown.append(item)

    output_error_dir = '../error_set/'
    error_PATH ='{}error_superior_{}_b_{}_e_{}.csv'.format(output_error_dir, USE_TAG, BATCH_SIZE, EPOCH)      
    with open(error_PATH, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(error_superior)

    output_error_dir = '../error_set/'
    error_PATH ='{}error_peer_{}_b_{}_e_{}.csv'.format(output_error_dir, USE_TAG, BATCH_SIZE, EPOCH)      
    with open(error_PATH, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(error_peer)

    output_error_dir = '../error_set/'
    error_PATH ='{}error_inferior_{}_b_{}_e_{}.csv'.format(output_error_dir, USE_TAG, BATCH_SIZE, EPOCH)      
    with open(error_PATH, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(error_inferior)

    output_error_dir = '../error_set/'
    error_PATH ='{}error_unknown_{}_b_{}_e_{}.csv'.format(output_error_dir, USE_TAG, BATCH_SIZE, EPOCH)      
    with open(error_PATH, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(error_unknown)
