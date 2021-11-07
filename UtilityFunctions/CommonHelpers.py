import pickle

def dump_pickle(file_name, obj):
    with open(file_name,"wb") as fi:
        pickle.dump(obj,fi)

def load_pickle(file_name):
    fi = open(file_name,"rb")
    obj = pickle.load(fi)
    return obj

def print_dict(dic):
    for key in dic:
        print(str(key)+"=>"+str(dic[key]))

def str_dict1(dic):
    ret = ""
    for key in dic:
        ret = ret+ str(key)+"=>"+str(dic[key])+","
    return ret