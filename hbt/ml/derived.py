# coding: utf-8

"""
ML models derived from the *SimpleDNN* class
"""


from hbt.ml.NN_1 import SimpleDNN


processes = [
    #"hh_ggf_bbtautau",
    # "hh_vbf_bbtautau",
    #"graviton_hh_ggf_bbtautau_m400",
    #"graviton_hh_vbf_bbtautau_m400",
    #"graviton_hh_ggf_bbtautau_m1250",
    # "graviton_hh_vbf_bbtautau_m1250",
    #"graviton_hh_ggf_bbtautau_m1500",
    # "graviton_hh_vbf_bbtautau_m1500",
    #"graviton_hh_ggf_bbtautau_m2000",
    # "graviton_hh_vbf_bbtautau_m2000",
]

ml_process_weights = {
    #"hh_ggf_bbtautau": 1,
    # "hh_vbf_bbtautau":1,
    #"graviton_hh_ggf_bbtautau_m400": 1,
    #"graviton_hh_vbf_bbtautau_m400": 1,
    #"graviton_hh_ggf_bbtautau_m1250": 1,
    # "graviton_hh_vbf_bbtautau_m1250": 1,
    #"graviton_hh_ggf_bbtautau_m1500": 1,
    # "graviton_hh_vbf_bbtautau_m1500": 1,
    #"graviton_hh_ggf_bbtautau_m2000": 1,
    # "graviton_hh_vbf_bbtautau_m2000": 1,
}

dataset_names = {
    #"ggHH_kl_1_kt_1_sl_hbbhww_powheg",
    # TTbar
    #"tt_sl_powheg",
    #"tt_dl_powheg",
    #"tt_fh_powheg",
    # SingleTop
    #"st_tchannel_t_powheg",
    #"st_tchannel_tbar_powheg",
    #"st_twchannel_t_powheg",
    #"st_twchannel_tbar_powheg",
    #"st_schannel_lep_amcatnlo",
    # "st_schannel_had_amcatnlo",
    # WJets
    #"w_lnu_ht70To100_madgraph",
    #"w_lnu_ht100To200_madgraph",
    #"w_lnu_ht200To400_madgraph",
    #"w_lnu_ht400To600_madgraph",
    #"w_lnu_ht600To800_madgraph",
    #"w_lnu_ht800To1200_madgraph",
    #"w_lnu_ht1200To2500_madgraph",
    #"w_lnu_ht2500_madgraph",
    # DY
    #"dy_lep_m50_ht70to100_madgraph",
    #"dy_lep_m50_ht100to200_madgraph",
    #"dy_lep_m50_ht200to400_madgraph",
    #"dy_lep_m50_ht400to600_madgraph",
    #"dy_lep_m50_ht600to800_madgraph",
    #"dy_lep_m50_ht800to1200_madgraph",
    #"dy_lep_m50_ht1200to2500_madgraph",
    #"dy_lep_m50_ht2500_madgraph",
    #"hh_ggf_bbtautau_madgraph",
    #"hh_vbf_bbtautau_madgraph",
    #"graviton_hh_ggf_bbtautau_m400_madgraph",
    #"graviton_hh_vbf_bbtautau_m400_madgraph",
    #"graviton_hh_ggf_bbtautau_m1250_madgraph",
    # "graviton_hh_vbf_bbtautau_m1250_madgraph",
    #"graviton_hh_ggf_bbtautau_m1500_madgraph",
    # "graviton_hh_vbf_bbtautau_m1500_madgraph",
    #"graviton_hh_ggf_bbtautau_m2000_madgraph",
    # "graviton_hh_vbf_bbtautau_m2000_madgraph",
}

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
    "folds": 3,
    # "max_events": 10**6,  # TODO
    "layers": [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 2],
    "activation": ["selu", "selu", "selu", "selu", "selu", "selu", "selu", "selu", "selu", "selu", "selu", 'softmax'],
    "l2_factor":0.01,
    "learningrate": 0.01,
    "batchsize": 131072,
    "epochs": 100,
    "eqweight": True,
    "dropout": 0.05,
    "processes": processes,
    "ml_process_weights": ml_process_weights,
    "dataset_names": dataset_names,
    "input_features": input_features,
    "store_name": "inputs1",
    "label_smoothing": 0,

    "early_stopping_patience":int(200/3),

    "scheduler_factor":0.5,
    "scheduler_patience":int(200/15),


    "cls_weight":{  0: 1,
                    1: 1,
                    #2: 11/7
    }
}

# derived model, usable on command line
default_dnn = SimpleDNN.derive("default", cls_dict=default_cls_dict)

# test model settings
cls_dict = default_cls_dict
cls_dict["epochs"] = 200
cls_dict["batchsize"] = 256
cls_dict["processes"] = [
    "graviton_hh_ggf_bbtautau_m400",
    "graviton_hh_vbf_bbtautau_m400",
]
cls_dict["dataset_names"] = {
    "graviton_hh_ggf_bbtautau_m400_madgraph",
    "graviton_hh_vbf_bbtautau_m400_madgraph",
}

test_dnn = SimpleDNN.derive("test", cls_dict=cls_dict)

#Combined model
combined_dict = default_cls_dict.copy()
combinedup = {
    "epochs": 200,
    "layers": [256, 256, 256, 256, 256, 256, 256, 1],
    "activation": ["elu", "elu", "elu", "elu", "elu", "elu", "elu", 'sigmoid'],
    "l2_factor": 0.01,
    "learningrate": 0.01,
    "folds": 4,
    "batchsize": 256,
    "early_stopping_patience":int(200/3),
    "scheduler_factor":0.5,
    "scheduler_patience":int(200/15),

    "processes": [
        "graviton_hh_ggf_bbtautau_m400",
        "graviton_hh_vbf_bbtautau_m400",
        "graviton_hh_ggf_bbtautau_m800",
        "graviton_hh_vbf_bbtautau_m800",
        "graviton_hh_ggf_bbtautau_m1250",
        "graviton_hh_vbf_bbtautau_m1250",
    ],
    "ml_process_weights":{
        "graviton_hh_ggf_bbtautau_m400": 1,
        "graviton_hh_vbf_bbtautau_m400": 1,
        "graviton_hh_ggf_bbtautau_m800": 1,
        "graviton_hh_vbf_bbtautau_m800": 1,
        "graviton_hh_ggf_bbtautau_m1250": 1,
        "graviton_hh_vbf_bbtautau_m1250": 1,
    },
    "dataset_names": {
        "graviton_hh_ggf_bbtautau_m400_madgraph",
        "graviton_hh_vbf_bbtautau_m400_madgraph",
        "graviton_hh_ggf_bbtautau_m800_madgraph",
        "graviton_hh_vbf_bbtautau_m800_madgraph",
        "graviton_hh_ggf_bbtautau_m1250_madgraph",
        "graviton_hh_vbf_bbtautau_m1250_madgraph",
    },
    "label": [
        '$HH_{ggf,m400,Graviton}$',
        '$HH_{vbf,m400,Graviton}$',
        '$HH_{ggf,m800,Graviton}$',
        '$HH_{vbf,m800,Graviton}$',
        '$HH_{ggf,m1250,Graviton}$',
        '$HH_{vbf,m1250,Graviton}$',
    ],

}
combined_dict.update(combinedup)

combined_dnn = SimpleDNN.derive("combined", cls_dict=combined_dict)

#Combined model
combined400only_dict = default_cls_dict.copy()
combined400onlyup = {
    "epochs": 200,
    "layers": [256, 256, 256, 256, 256, 256, 256, 1],
    "activation": ["elu", "elu", "elu", "elu", "elu", "elu", "elu", 'sigmoid'],
    "l2_factor": 0.01,
    "learningrate": 0.01,
    "folds": 4,
    "batchsize": 256,
    "early_stopping_patience":int(200/3),
    "scheduler_factor":0.5,
    "scheduler_patience":int(200/15),
    "processes": [
        "graviton_hh_ggf_bbtautau_m400",
        "graviton_hh_vbf_bbtautau_m400",

    ],
    "ml_process_weights":{
        "graviton_hh_ggf_bbtautau_m400": 1,
        "graviton_hh_vbf_bbtautau_m400": 1,

    },
    "dataset_names": {
        "graviton_hh_ggf_bbtautau_m400_madgraph",
        "graviton_hh_vbf_bbtautau_m400_madgraph",

    },
    "label": [
        '$HH_{ggf,m400,Graviton}$',
        '$HH_{vbf,m400,Graviton}$',

    ],

}
combined400only_dict.update(combined400onlyup)

combined400only_dnn = SimpleDNN.derive("combined400only", cls_dict=combined400only_dict)

#m400 model
m400_dict = default_cls_dict.copy()
m400up = {
    "epochs": 200,
    "layers": [256, 256, 256, 256, 256, 256, 256, 2],
    "activation": ["elu", "elu", "elu", "elu", "elu", "elu", "elu", 'softmax'],
    "l2_factor": 0.01,
    "learningrate": 0.01,
    "folds": 4,
    "batchsize": 256,
    "early_stopping_patience":int(200/3),
    "scheduler_factor":0.5,
    "scheduler_patience":int(200/15),
    "cls_weight": {
        0: 1,
        1: 1
    },
    "processes": [
        "graviton_hh_ggf_bbtautau_m400",
        "graviton_hh_vbf_bbtautau_m400",
    ],
    "ml_process_weights":{
        "graviton_hh_ggf_bbtautau_m400": 1,
        "graviton_hh_vbf_bbtautau_m400": 1,
    },
    "dataset_names": {
        "graviton_hh_ggf_bbtautau_m400_madgraph",
        "graviton_hh_vbf_bbtautau_m400_madgraph",
    },
    "label": [
        '$HH_{ggf,m400,Graviton}$',
        '$HH_{vbf,m400,Graviton}$',
    ],

}
m400_dict.update(m400up)

m400_dnn = SimpleDNN.derive("m400", cls_dict=m400_dict)

#m400best model
m400best_dict = default_cls_dict.copy()
m400bestup = {
    "epochs": 200,
    "layers": [256, 256, 256, 256, 256, 256, 256, 2],
    "activation": ["elu", "elu", "elu", "elu", "elu", "elu", "elu", 'softmax'],
    "l2_factor": 0.01,
    "learningrate": 0.01,
    "folds": 4,
    "batchsize": 256,
    "early_stopping_patience":int(200/3),
    "scheduler_factor":0.5,
    "scheduler_patience":int(200/15),
    "cls_weight": {
        0: 1,
        1: 1
    },
    "processes": [
        "graviton_hh_ggf_bbtautau_m400",
        "graviton_hh_vbf_bbtautau_m400",
    ],
    "ml_process_weights":{
        "graviton_hh_ggf_bbtautau_m400": 1,
        "graviton_hh_vbf_bbtautau_m400": 1,
    },
    "dataset_names": {
        "graviton_hh_ggf_bbtautau_m400_madgraph",
        "graviton_hh_vbf_bbtautau_m400_madgraph",
    },
    "label": [
        '$HH_{ggf,m400,Graviton}$',
        '$HH_{vbf,m400,Graviton}$',
    ],

}
m400best_dict.update(m400bestup)

m400best_dnn = SimpleDNN.derive("m400best", cls_dict=m400best_dict)


#m400mid model
m400mid_dict = default_cls_dict.copy()
m400midup = {
    "epochs": 200,
    "layers": [64, 64, 64, 64, 64, 2],
    "activation": ["elu", "elu", "elu", "elu", "elu", 'softmax'],
    "l2_factor": 0.01,
    "learningrate": 0.01,
    "folds": 4,
    "batchsize": 256,
    "early_stopping_patience":int(200/3),
    "scheduler_factor":0.5,
    "scheduler_patience":int(200/15),
    "cls_weight": {
        0: 1,
        1: 1
    },
    "processes": [
        "graviton_hh_ggf_bbtautau_m400",
        "graviton_hh_vbf_bbtautau_m400",
    ],
    "ml_process_weights":{
        "graviton_hh_ggf_bbtautau_m400": 1,
        "graviton_hh_vbf_bbtautau_m400": 1,
    },
    "dataset_names": {
        "graviton_hh_ggf_bbtautau_m400_madgraph",
        "graviton_hh_vbf_bbtautau_m400_madgraph",
    },
    "label": [
        '$HH_{ggf,m400,Graviton}$',
        '$HH_{vbf,m400,Graviton}$',
    ],

}
m400mid_dict.update(m400midup)

m400mid_dnn = SimpleDNN.derive("m400mid", cls_dict=m400mid_dict)


#m400low model
m400low_dict = default_cls_dict.copy()
m400lowup = {
    "epochs": 200,
    "layers": [128, 128, 128, 128, 128, 128, 128, 128, 128, 2],
    "activation": ["elu", "elu", "elu", "elu", "elu", "elu", "elu", "elu", "elu", 'softmax'],
    "l2_factor": 0.02,
    "learningrate": 0.1,
    "folds": 4,
    "batchsize": 256,
    "early_stopping_patience":int(200/3),
    "scheduler_factor":0.5,
    "scheduler_patience":int(200/15),
    "cls_weight": {
        0: 1,
        1: 1
    },
    "processes": [
        "graviton_hh_ggf_bbtautau_m400",
        "graviton_hh_vbf_bbtautau_m400",
    ],
    "ml_process_weights":{
        "graviton_hh_ggf_bbtautau_m400": 1,
        "graviton_hh_vbf_bbtautau_m400": 1,
    },
    "dataset_names": {
        "graviton_hh_ggf_bbtautau_m400_madgraph",
        "graviton_hh_vbf_bbtautau_m400_madgraph",
    },
    "label": [
        '$HH_{ggf,m400,Graviton}$',
        '$HH_{vbf,m400,Graviton}$',
    ],

}
m400low_dict.update(m400lowup)

m400low_dnn = SimpleDNN.derive("m400low", cls_dict=m400low_dict)


#m1250 model
m1250_dict = default_cls_dict.copy()
m1250up = {
    "epochs": 200,
    "layers": [128, 128, 128, 128, 2],
    "activation": ["selu", "selu", "selu", "selu", 'softmax'],
    "l2_factor": 0.01,
    "learningrate": 0.01,
    "folds": 4,
    "batchsize": 256,
    "early_stopping_patience":int(200/3),
    "scheduler_factor":0.5,
    "scheduler_patience":int(200/15),
    "cls_weight": {
        0: 1,
        1: 1.8
    },
    "processes": [
        "graviton_hh_ggf_bbtautau_m1250",
        "graviton_hh_vbf_bbtautau_m1250",
    ],
    "ml_process_weights":{
        "graviton_hh_ggf_bbtautau_m1250": 1,
        "graviton_hh_vbf_bbtautau_m1250": 1,
    },
    "dataset_names": {
        "graviton_hh_ggf_bbtautau_m1250_madgraph",
        "graviton_hh_vbf_bbtautau_m1250_madgraph",
    },
        "label": [
        '$HH_{ggf,m1250,Graviton}$',
        '$HH_{vbf,m1250,Graviton}$',
    ],

}
m1250_dict.update(m1250up)

m1250_dnn = SimpleDNN.derive("m1250", cls_dict=m1250_dict)

#m1250best model
m1250best_dict = default_cls_dict.copy()
m1250bestup = {
    "epochs": 200,
    "layers": [256, 256, 256, 256, 256, 256, 256, 2],
    "activation": ["elu", "elu", "elu", "elu", "elu", "elu", "elu", 'softmax'],
    "l2_factor": 0.01,
    "learningrate": 0.01,
    "folds": 4,
    "batchsize": 256,
    "early_stopping_patience":int(200/3),
    "scheduler_factor":0.5,
    "scheduler_patience":int(200/15),
    "cls_weight": {
        0: 1,
        1: 1.8
    },
    "processes": [
        "graviton_hh_ggf_bbtautau_m1250",
        "graviton_hh_vbf_bbtautau_m1250",
    ],
    "ml_process_weights":{
        "graviton_hh_ggf_bbtautau_m1250": 1,
        "graviton_hh_vbf_bbtautau_m1250": 1,
    },
    "dataset_names": {
        "graviton_hh_ggf_bbtautau_m1250_madgraph",
        "graviton_hh_vbf_bbtautau_m1250_madgraph",
    },
    "label": [
        '$HH_{ggf,m1250,Graviton}$',
        '$HH_{vbf,m1250,Graviton}$',
    ],

}
m1250best_dict.update(m1250bestup)

m1250best_dnn = SimpleDNN.derive("m1250best", cls_dict=m1250best_dict)


#m1250mid model
m1250mid_dict = default_cls_dict.copy()
m1250midup = {
    "epochs": 200,
    "layers": [64, 64, 64, 64, 64, 2],
    "activation": ["elu", "elu", "elu", "elu", "elu", 'softmax'],
    "l2_factor": 0.01,
    "learningrate": 0.01,
    "folds": 4,
    "batchsize": 256,
    "early_stopping_patience":int(200/3),
    "scheduler_factor":0.5,
    "scheduler_patience":int(200/15),
    "cls_weight": {
        0: 1,
        1: 1.8
    },
    "processes": [
        "graviton_hh_ggf_bbtautau_m1250",
        "graviton_hh_vbf_bbtautau_m1250",
    ],
    "ml_process_weights":{
        "graviton_hh_ggf_bbtautau_m1250": 1,
        "graviton_hh_vbf_bbtautau_m1250": 1,
    },
    "dataset_names": {
        "graviton_hh_ggf_bbtautau_m1250_madgraph",
        "graviton_hh_vbf_bbtautau_m1250_madgraph",
    },
    "label": [
        '$HH_{ggf,m1250,Graviton}$',
        '$HH_{vbf,m1250,Graviton}$',
    ],

}
m1250mid_dict.update(m1250midup)

m1250mid_dnn = SimpleDNN.derive("m1250mid", cls_dict=m1250mid_dict)


#m1250low model
m1250low_dict = default_cls_dict.copy()
m1250lowup = {
    "epochs": 200,
    "layers": [128, 128, 128, 128, 128, 128, 128, 128, 128, 2],
    "activation": ["elu", "elu", "elu", "elu", "elu", "elu", "elu", "elu", "elu", 'softmax'],
    "l2_factor": 0.02,
    "learningrate": 0.1,
    "folds": 4,
    "batchsize": 256,
    "early_stopping_patience":int(200/3),
    "scheduler_factor":0.5,
    "scheduler_patience":int(200/15),
    "cls_weight": {
        0: 1,
        1: 1.8
    },
    "processes": [
        "graviton_hh_ggf_bbtautau_m1250",
        "graviton_hh_vbf_bbtautau_m1250",
    ],
    "ml_process_weights":{
        "graviton_hh_ggf_bbtautau_m1250": 1,
        "graviton_hh_vbf_bbtautau_m1250": 1,
    },
    "dataset_names": {
        "graviton_hh_ggf_bbtautau_m1250_madgraph",
        "graviton_hh_vbf_bbtautau_m1250_madgraph",
    },
    "label": [
        '$HH_{ggf,m1250,Graviton}$',
        '$HH_{vbf,m1250,Graviton}$',
    ],

}
m1250low_dict.update(m1250lowup)

m1250low_dnn = SimpleDNN.derive("m1250low", cls_dict=m1250low_dict)

#m800 model
m800_dict = default_cls_dict.copy()
m800up = {
    "epochs": 200,
    "layers": [128, 128, 128, 128, 2],
    "activation": ["selu", "selu", "selu", "selu", 'softmax'],
    "l2_factor": 0.01,
    "learningrate": 0.01,
    "folds": 4,
    "batchsize": 256,
    "early_stopping_patience":int(200/3),
    "scheduler_factor":0.5,
    "scheduler_patience":int(200/15),
    "cls_weight": {
        0: 1,
        1: 10
    },
    "processes": [
        "graviton_hh_ggf_bbtautau_m800",
        "graviton_hh_vbf_bbtautau_m800",
    ],
    "ml_process_weights":{
        "graviton_hh_ggf_bbtautau_m800": 1,
        "graviton_hh_vbf_bbtautau_m800": 1,
    },
    "dataset_names": {
        "graviton_hh_ggf_bbtautau_m800_madgraph",
        "graviton_hh_vbf_bbtautau_m800_madgraph",
    },
        "label": [
        '$HH_{ggf,m800,Graviton}$',
        '$HH_{vbf,m800,Graviton}$',
    ],

}
m800_dict.update(m800up)

m800_dnn = SimpleDNN.derive("m800", cls_dict=m800_dict)

#m800best model
m800best_dict = default_cls_dict.copy()
m800bestup = {
    "epochs": 200,
    "layers": [256, 256, 256, 256, 256, 256, 256, 2],
    "activation": ["elu", "elu", "elu", "elu", "elu", "elu", "elu", 'softmax'],
    "l2_factor": 0.01,
    "learningrate": 0.01,
    "folds": 4,
    "batchsize": 256,
    "early_stopping_patience":int(200/3),
    "scheduler_factor":0.5,
    "scheduler_patience":int(200/15),
    "cls_weight": {
        0: 1,
        1: 10
    },
    "processes": [
        "graviton_hh_ggf_bbtautau_m800",
        "graviton_hh_vbf_bbtautau_m800",
    ],
    "ml_process_weights":{
        "graviton_hh_ggf_bbtautau_m800": 1,
        "graviton_hh_vbf_bbtautau_m800": 1,
    },
    "dataset_names": {
        "graviton_hh_ggf_bbtautau_m800_madgraph",
        "graviton_hh_vbf_bbtautau_m800_madgraph",
    },
    "label": [
        '$HH_{ggf,m800,Graviton}$',
        '$HH_{vbf,m800,Graviton}$',
    ],

}
m800best_dict.update(m800bestup)

m800best_dnn = SimpleDNN.derive("m800best", cls_dict=m800best_dict)


#m800mid model
m800mid_dict = default_cls_dict.copy()
m800midup = {
    "epochs": 200,
    "layers": [64, 64, 64, 64, 64, 2],
    "activation": ["elu", "elu", "elu", "elu", "elu", 'softmax'],
    "l2_factor": 0.01,
    "learningrate": 0.01,
    "folds": 4,
    "batchsize": 256,
    "early_stopping_patience":int(200/3),
    "scheduler_factor":0.5,
    "scheduler_patience":int(200/15),
    "cls_weight": {
        0: 1,
        1: 10
    },
    "processes": [
        "graviton_hh_ggf_bbtautau_m800",
        "graviton_hh_vbf_bbtautau_m800",
    ],
    "ml_process_weights":{
        "graviton_hh_ggf_bbtautau_m800": 1,
        "graviton_hh_vbf_bbtautau_m800": 1,
    },
    "dataset_names": {
        "graviton_hh_ggf_bbtautau_m800_madgraph",
        "graviton_hh_vbf_bbtautau_m800_madgraph",
    },
    "label": [
        '$HH_{ggf,m800,Graviton}$',
        '$HH_{vbf,m800,Graviton}$',
    ],

}
m800mid_dict.update(m800midup)

m800mid_dnn = SimpleDNN.derive("m800mid", cls_dict=m800mid_dict)


#m800low model
m800low_dict = default_cls_dict.copy()
m800lowup = {
    "epochs": 200,
    "layers": [128, 128, 128, 128, 128, 128, 128, 128, 128, 2],
    "activation": ["elu", "elu", "elu", "elu", "elu", "elu", "elu", "elu", "elu", 'softmax'],
    "l2_factor": 0.02,
    "learningrate": 0.1,
    "folds": 4,
    "batchsize": 256,
    "early_stopping_patience":int(200/3),
    "scheduler_factor":0.5,
    "scheduler_patience":int(200/15),
    "cls_weight": {
        0: 1,
        1: 10
    },
    "processes": [
        "graviton_hh_ggf_bbtautau_m800",
        "graviton_hh_vbf_bbtautau_m800",
    ],
    "ml_process_weights":{
        "graviton_hh_ggf_bbtautau_m800": 1,
        "graviton_hh_vbf_bbtautau_m800": 1,
    },
    "dataset_names": {
        "graviton_hh_ggf_bbtautau_m800_madgraph",
        "graviton_hh_vbf_bbtautau_m800_madgraph",
    },
    "label": [
        '$HH_{ggf,m800,Graviton}$',
        '$HH_{vbf,m800,Graviton}$',
    ],

}
m800low_dict.update(m800lowup)

m800low_dnn = SimpleDNN.derive("m800low", cls_dict=m800low_dict)