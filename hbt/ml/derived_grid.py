# coding: utf-8

"""
ML models derived from the *SimpleDNN* class
"""

# from columnflow.util import maybe_import
from hbt.ml.NN_1 import SimpleDNN
from hbt.ml import grid_config

processes = [
        "graviton_hh_ggf_bbtautau_m400",
        "graviton_hh_vbf_bbtautau_m400",
]

ml_process_weights = {
        "graviton_hh_ggf_bbtautau_m400": 1,
        "graviton_hh_vbf_bbtautau_m400": 1,
}

dataset_names = {
        "graviton_hh_ggf_bbtautau_m400_madgraph",
        "graviton_hh_vbf_bbtautau_m400_madgraph",
}

label = [
        '$HH_{ggf,m400,Graviton}$',
        '$HH_{vbf,m400,Graviton}$',
]


input_features = [
    f"{obj}_{var}"
    for obj in ["jet1", "jet2", "jet3", "jet4", "bjet1", "bjet2", "bjet3", "bjet4", "tau1", "tau2", "vbfjet1", "vbfjet2"]
    for var in ["p", "pt", "eta", "phi", "mass", "e"]] + ["mtautau", "mjj", "mbjetbjet", "mHH"]
    
    # f"{obj}_{var}"
    # for obj in ["jet1", "jet2"]
    # for var in ["area", "nConstituents", "hadronFlavour"]] + [
    # f"{obj}_{var}"
    # for obj in ["bjet1", "bjet2"]
    # for var in ["area", "nConstituents", "btag"]] + ["jets_nJets", "bjets_nJets"]


default_cls_dict = {
    "folds": 4,
    # "max_events": 10**6,  # TODO
    # "activation": ["selu", "selu", "selu", "selu", "selu", "selu", "selu", "selu", "selu", "selu", "selu", 'softmax'],
    "batchsize": 256,
    "epochs": 200,
    "eqweight": True,
    "processes": processes,
    "ml_process_weights": ml_process_weights,
    "dataset_names": dataset_names,
    "input_features": input_features,
    "label": label,
    "store_name": "inputs1",
    "label_smoothing":0.1,

    "early_stopping_patience":int(200/3),

    "scheduler_factor":0.5,
    "scheduler_patience":int(200/15),


    "cls_weight":{  0: 1,
                    1: 1,
                    #2: 11/7
    }
}
model_name = "m400"
grid = grid_config.create_iterproduct_of_config(config=grid_config.CONFIG,
                                                base_config=default_cls_dict,
                                                model_name=model_name)
# if callable(grid_config.create_iterproduct_of_config):
#         grid = grid_config.create_iterproduct_of_config(config=grid_config.CONFIG,
#                                                 base_config=default_cls_dict,
#                                                 model_name=model_name)
# else:
#       grid=dict()
all_derived_models = []
for model_name in grid.keys():
    model_config = grid[model_name]
    model_config.update(default_cls_dict)
    
    derived_model = SimpleDNN.derive(model_name, cls_dict=model_config)
    all_derived_models.append(derived_model)
