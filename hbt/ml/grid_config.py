import itertools
from pathlib import Path
# config for all configs
BASE_CONFIG = {
    'num_classes':2,
}

default_cls_dict = {
    "folds": 5,
    # "max_events": 10**6,  # TODO
    # "activation": ["selu", "selu", "selu", "selu", "selu", "selu", "selu", "selu", "selu", "selu", "selu", 'softmax'],
    "batchsize": 256,
    "epochs": 200,
    "eqweight": True,
    # "processes": processes,
    # "ml_process_weights": ml_process_weights,
    # "dataset_names": dataset_names,
    # "input_features": input_features,
    "store_name": "inputs1",
    "label_smoothing":0.1,

    "early_stopping_patience":int(200/3),

    "scheduler_factor":0.5,
    "scheduler_patience":int(200/15),
    "dataset_names": ("A","B"),


    "cls_weight":{  0: 1,
                    1: 1,
                    #2: 11/7
    }
}



# new config combinations
CONFIG = {
'nodes':[64,128,256],
'depth':[5,7,9],
'l2_factor':[0.01,0.02,0.005],
'learningrate':[0.01,0.1,0.05],
'activation_function':['elu'],
'batch_size':[256]
}


def create_string(dict_parameter, model_name):
    d = dict_parameter
    def replace_dots(value):
        string = str(value)
        return string.replace(".","-")

    return f"{model_name}_N{d['nodes']}_D{d['depth']}_LR{replace_dots(d['learningrate'])}_L2F{replace_dots(d['l2_factor'])}_BS{d['batch_size']}_ACT{d['activation_function']}"


def remove(dict_parameter, to_delete=('nodes', 'depth')):
    d = dict_parameter
    # remove nodes, depth
    to_delete = to_delete
    for delete_key in to_delete:
        del d[delete_key]


def create_iterproduct_of_config(config, base_config,model_name, save_as_text=""):
    final_dict = {}
    config_keys = config.keys()
    for values in itertools.product(*config.values()):

        config_setting_per_network = dict(zip(config_keys, values))

        layer = [config_setting_per_network["nodes"] for num_layer in range(config_setting_per_network["depth"])]
        layer.append(len(base_config['dataset_names']))

        activation_functions = [CONFIG["activation_function"][0] for node in layer[:-1]]
        activation_functions.append("Softmax")

        config_setting_per_network['layers'] = layer
        config_setting_per_network['activation'] = activation_functions

        hyperparameter_key_model_name = create_string(dict_parameter=config_setting_per_network, model_name=model_name)
        
        # remove unnecessary parameter
        remove(config_setting_per_network, ('nodes', 'depth', 'activation_function'))

        final_dict[hyperparameter_key_model_name] = config_setting_per_network

        if save_as_text:
            p = Path(save_as_text)
            with p.open("w") as file:
                for line in final_dict.keys():
                    file.write(f"{line}\n")
    return final_dict


if __name__ == "__main__":

    model_name = "m400"
    save_as_text = "/home/lebenjam/afs/code/hh2bbtautau/hbt/ml/grid_{model_name}.config"

    dummy_config = create_iterproduct_of_config(CONFIG,default_cls_dict, model_name, save_as_text=save_as_text)
    print(*dummy_config.values(), sep='\n')