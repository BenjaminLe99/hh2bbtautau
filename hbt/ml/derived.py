# coding: utf-8

"""
ML models derived from the *SimpleDNN* class
"""


from hbt.ml.NN_1 import SimpleDNN


processes = [
    "hh_ggf_bbtautau",
    # "hh_vbf_bbtautau",
    "graviton_hh_ggf_bbtautau_m400",
    "graviton_hh_vbf_bbtautau_m400",
    #"graviton_hh_ggf_bbtautau_m1250",
    # "graviton_hh_vbf_bbtautau_m1250",
    #"graviton_hh_ggf_bbtautau_m1500",
    # "graviton_hh_vbf_bbtautau_m1500",
    #"graviton_hh_ggf_bbtautau_m2000",
    # "graviton_hh_vbf_bbtautau_m2000",
]

ml_process_weights = {
    "hh_ggf_bbtautau": 1,
    # "hh_vbf_bbtautau":1,
    "graviton_hh_ggf_bbtautau_m400": 1,
    "graviton_hh_vbf_bbtautau_m400": 1,
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
    "hh_ggf_bbtautau_madgraph",
    #"hh_vbf_bbtautau_madgraph",
    "graviton_hh_ggf_bbtautau_m400_madgraph",
    "graviton_hh_vbf_bbtautau_m400_madgraph",
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
    "layers": [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 3],
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
    "label_smoothing":0.1,

    "early_stopping_patience":int(20 / 3),

    "scheduler_factor":0.5,
    "scheduler_patience":int(20/15),


    "cls_weight":{  0: 1,
                    1: 11/7,
                    2: 11/7}
}


# derived model, usable on command line
default_dnn = SimpleDNN.derive("default", cls_dict=default_cls_dict)

# test model settings
cls_dict = default_cls_dict
cls_dict["epochs"] = 20
cls_dict["batchsize"] = 256
cls_dict["processes"] = [
    "hh_ggf_bbtautau",
    # "hh_vbf_bbtautau",
    "graviton_hh_ggf_bbtautau_m400",
    "graviton_hh_vbf_bbtautau_m400",
    #"graviton_hh_ggf_bbtautau_m1250",
    # "graviton_hh_vbf_bbtautau_m1250",
    #"graviton_hh_ggf_bbtautau_m1500",
    # "graviton_hh_vbf_bbtautau_m1500",
    #"graviton_hh_ggf_bbtautau_m2000",
    # "graviton_hh_vbf_bbtautau_m2000",
]
cls_dict["dataset_names"] = {
    "hh_ggf_bbtautau_madgraph",
    #"hh_vbf_bbtautau_madgraph",
    "graviton_hh_ggf_bbtautau_m400_madgraph",
    "graviton_hh_vbf_bbtautau_m400_madgraph",
    #"graviton_hh_ggf_bbtautau_m1250_madgraph",
    # "graviton_hh_vbf_bbtautau_m1250_madgraph",
    #"graviton_hh_ggf_bbtautau_m1500_madgraph",
    # "graviton_hh_vbf_bbtautau_m1500_madgraph",
    #"graviton_hh_ggf_bbtautau_m2000_madgraph",
    # "graviton_hh_vbf_bbtautau_m2000_madgraph",
}

test_dnn = SimpleDNN.derive("test", cls_dict=cls_dict)