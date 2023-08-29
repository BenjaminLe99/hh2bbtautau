import pathlib
import json
import numpy as np


HBT_STORE = "/nfs/dust/cms/user/lebenjam/columnflow/data/hbt_store/analysis_hbt/cf.MLTraining"

def get_ml_names(file_path):
    p = pathlib.Path(file_path)
    file = p.open("r")
    ml_names = [f"ml__{line}".replace("\n","") for line in file.readlines()]
    return ml_names

def full_path(ml_name, version):
    parts = (HBT_STORE,ml_name,str(version))
    return pathlib.Path().joinpath(*parts)

def get_metric_of_model(path, fold):
    
    patter = f"mlmodel_f*of{fold}/*.json"
    data = {}
    for file in pathlib.Path(path).glob(patter):
        parts = file.parts

        fold_name = parts[-2]
        json_file = file.open("r").read()
        # is dictionary
        metric_value = json.loads(json_file)

        # if not placed, set dictionary, else update
        if fold_name not in data:
            data[fold_name] = {}
        
        data[fold_name].update(metric_value)
        
    return data

def gather_metrics(config,version,fold):
    all_ml_models = get_ml_names(config)
    
    all_data = {}
    for model in all_ml_models:
        f_path = full_path(model,version)
        all_data[model] = get_metric_of_model(f_path,fold)

    return all_data

def average_metrics(metric_dict, max_epoch=199):
    averaged_model_metrics = {}
    for ml_model in metric_dict:
        if ml_model not in averaged_model_metrics.keys():
            averaged_model_metrics[ml_model] = {}

        auc_scores = []
        train_acc = []
        valid_acc = []

        metric_ml_model = metric_dict[ml_model]

        for fold_name in metric_ml_model:
            fold = metric_ml_model[fold_name]

            auc_scores.append(fold["auc_score"])
            if fold["early_stopped_epoch"] == 0:
                fold["early_stopped_epoch"] = max_epoch

            train_acc.append(fold["training_categorial_accuracy"][fold["early_stopped_epoch"]])
            valid_acc.append(fold["validation_categorial_accuracy"][fold["early_stopped_epoch"]])

        for score_name, score_value in zip(("auc_score","trainings_accuracy","validation_accuracy"),(auc_scores, train_acc, valid_acc)):
            mean = np.mean(score_value)
            std = np.std(score_value)
            
            averaged_model_metrics[ml_model][score_name] = {}
            averaged_model_metrics[ml_model][score_name]["mean"] = mean
            averaged_model_metrics[ml_model][score_name]["std"] = std
    #print(averaged_model_metrics)
    return averaged_model_metrics

def get_highest_scores(averaged_metrics):
    all_scores = []
    for ml_model_name, scores in averaged_metrics.items():
        auc_score = scores["auc_score"]
        train_acc = scores["trainings_accuracy"]
        val_acc = scores["validation_accuracy"]

        all_scores.append((ml_model_name, auc_score["mean"], auc_score["std"], train_acc["mean"], train_acc["std"], val_acc["mean"], val_acc["std"]))

    # sort values after mean value (second entry)
    sorted_auc_mean = sorted(all_scores, key=lambda x: x[1])
    sorted_train_mean = sorted(all_scores, key=lambda x: x[3])
    sorted_val_mean = sorted(all_scores, key=lambda x: x[5])

    return sorted_auc_mean, sorted_train_mean, sorted_val_mean

if __name__ == "__main__":
    config_path = "/afs/desy.de/user/l/lebenjam/code/hh2bbtautau/hbt/ml/grid_config_m400.txt"
    VERSION=1
    FOLD = 4

    metrics = gather_metrics(config_path,VERSION,FOLD)
    #print(average_metrics(metrics,199))
    auc_mean, train_mean, val_mean = get_highest_scores(average_metrics(metrics, 199))
    print(*auc_mean, sep="\n")
    print("----------------------------------------")
    print(*auc_mean[-4:], sep="\n")
    print(*train_mean[-4:], sep="\n")
    print(*val_mean[-4:], sep="\n")
    
    
    def print_confusion_plot_path(metric, verbose=None):
        if verbose:
            print(f"Metric is:{verbose}")
        confusion_validation_path="/nfs/dust/cms/user/lebenjam/columnflow/data/hbt_store/analysis_hbt/cf.MLTraining/{ml_name}/1/mlmodel_f0of4/Confusion_validation.pdf"

        best_model = []
        for model in metric[-4:]:
            p = confusion_validation_path.format(ml_name=model[0])
            best_model.append(p)
        print(*best_model, sep="\n")
    
    print_confusion_plot_path(auc_mean, "Auc-Score")
    print_confusion_plot_path(train_mean, "Train Accuracy")
    print_confusion_plot_path(val_mean, "Val Accuracy")
