import torch as t


def check_equals(dict1, dict2):
    flag = True
    for name in dict1.keys():
        if name in dict1 and name in dict2:
            if not t.equal(dict1[name].cpu(), dict2[name].cpu()):
                print(name)
                flag = False
    return flag


def check_equals_bn(dict1, dict2):
    flag = True
    for key in dict2.keys():
        if "running" not in key:
            if not t.equal(dict1[key].cpu(), dict2[key].cpu()):
                flag = False
                print(key)
    return flag
