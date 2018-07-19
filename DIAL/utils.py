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
    for val in dict2.values():
        val.cpu()
    for key in dict2.keys():
        if "bn" in key:
            if "running_mean" in key or "running_var" in key:
                continue
            if "weight" in key:
                if not(t.equal(dict1[key[:-6] + "scale.weight"], dict2[key].cpu())) :
                    print(key)
            elif "bias" in key:
                if not (t.equal(dict1[key[:-4] + "scale.bias"], dict2[key].cpu())):
                    print(key)
        elif 'downsample' in key:
            if "running_mean" in key or "running_var" in key:
                continue
            if "0.weight" in key or "0.bias" in key:
                if not t.equal(dict1[key], dict2[key].cpu()):
                    print(key)
            elif "1.weight" in key:
                if not (t.equal(dict1[key[:-6] + "scale.weight"], dict2[key].cpu())):
                    print(key)
            elif "1.bias" in key:
                if not (t.equal(dict1[key[:-4] + "scale.bias"], dict2[key].cpu())):
                    print(key)
            else:
                if not t.equal(dict1[key], dict2[key]):
                    print(key)
    return flag