set_dict = {
    "def": {
        "epochs": 60,
        "lr": 0.001,
        "step": 40,
        "decay": 0.00001,
        "bs": 64,
        "adamlr": 0.00001
        },
    "rod": {
        "epochs": 30,
        "lr": 0.005,
        "step": 20,
        "decay": 0.00005,
        "bs": 32,
        "adamlr": 0.00001
    }
}


def get_parameter_set(name):
    return set_dict[name]
