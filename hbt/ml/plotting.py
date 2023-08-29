# coding: utf-8

from __future__ import annotations

import law
import order as od

from columnflow.util import maybe_import, DotDict

shap = maybe_import("shap")
np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
hist = maybe_import("hist")
tf = maybe_import("tensorflow")
sklearn = maybe_import("sklearn")


def plot_loss(history, output) -> None:
    """
    Simple function to create and store a loss plot
    """
    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)

    fig, ax = plt.subplots()
    trainings_loss = history["loss"]
    validation_loss = history["val_loss"]
    ax.plot(trainings_loss)
    ax.plot(validation_loss)
    ax.set(**{
        "ylabel": "Loss",
        "xlabel": "Epoch",
    })
    ax.legend(["train", "validation"], loc="best")
    mplhep.cms.label(ax=ax, llabel="Work in progress", data=False)

    output.child("Loss.pdf", type="f").dump(fig, formatter="mpl")

    json_metric = {"training_loss":trainings_loss, "validation_loss":validation_loss}
    output.child("loss.json", type="f").dump(json_metric, formatter="json")


def plot_accuracy(history, output) -> None:
    """
    Simple function to create and store an accuracy plot
    """
    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)

    fig, ax = plt.subplots()

    categorial_acc = history["categorical_accuracy"]
    validation_categorial_acc = history["val_categorical_accuracy"]
    ax.plot(categorial_acc)
    ax.plot(validation_categorial_acc)
    ax.set(**{
        "ylabel": "Accuracy",
        "xlabel": "Epoch",
    })
    ax.legend(["train", "validation"], loc="best")
    mplhep.cms.label(ax=ax, llabel="Work in progress", data=False)

    output.child("Accuracy.pdf", type="f").dump(fig, formatter="mpl")


    # save metric as json
    json_metric = {"training_categorial_accuracy":categorial_acc, "validation_categorial_accuracy":validation_categorial_acc}
    json_path = "accuracy.json"
    output.child(json_path, type="f").dump(json_metric, formatter="json")


def plot_confusion(
        model: tf.keras.models.Model,
        inputs: DotDict,
        output: law.FileSystemDirectoryTarget,
        input_type: str,
        process_insts: tuple[od.Process],
        label=None
) -> None:
    """
    Simple function to create and store a confusion matrix plot
    """
    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Create confusion matrix and normalizes it over predicted (columns)
    confusion = confusion_matrix(
        y_true=np.argmax(inputs["target"], axis=-1),
        y_pred=np.argmax(inputs["prediction"], axis=-1),
        sample_weight=inputs["weights"],
        normalize="true",
    )


    #labels = [proc_inst.label for proc_inst in process_insts] if process_insts else None
    #from IPython import embed; embed();

    # Create a plot of the confusion matrix
    fig, ax = plt.subplots()
    matrix_display = ConfusionMatrixDisplay(confusion, display_labels=label)
    matrix_display.plot(ax=ax)
    matrix_display.im_.set_clim(0, 1)
    #ConfusionMatrixDisplay(confusion, display_labels=labels).plot(ax=ax)

    ax.set_title(f"Confusion matrix for {input_type} set, rows normalized", fontsize=20)
    mplhep.cms.label(ax=ax, llabel="Work in progress", data=False, loc=2)

    output.child(f"Confusion_{input_type}.pdf", type="f").dump(fig, formatter="mpl")


def plot_roc_ovr(
        model: tf.keras.models.Model,
        inputs: DotDict,
        output: law.FileSystemDirectoryTarget,
        input_type: str,
        process_insts: tuple[od.Process],
        label=None
) -> None:
    """
    Simple function to create and store some ROC plots;
    mode: OvR (one versus rest)
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    #from IPython import embed; embed();
    auc_scores = []
    n_classes = inputs["target"].shape[-1]

    fig, ax = plt.subplots()
    for i in range(n_classes):
        fpr, tpr, thresholds = roc_curve(
            y_true=inputs["target"][:,i],
            y_score=inputs["prediction"][:,i],
            sample_weight=inputs["weights"],
        )

        auc_scores.append(roc_auc_score(
            inputs["target"][:,i],
            inputs["prediction"][:,i],
            average="macro", multi_class="ovr",
        ))

        # create the plot
        ax.plot(fpr, tpr)

    ax.set_title(f"ROC OvR, {input_type} set")
    ax.set_xlabel("Background selection efficiency (FPR)")
    ax.set_ylabel("Signal selection efficiency (TPR)")

    # legend
    labels = label
    auc_legend = []
    for i, auc_score in enumerate(auc_scores):
        legend_label = f"Signal:{labels[i]} (AUC:{auc_score:.4f})"
        auc_legend.append(legend_label)

    ax.legend(auc_legend,loc="best")
    mplhep.cms.label(ax=ax, llabel="Work in progress", data=False, loc=2)

    output.child(f"ROC_ovr_{input_type}.pdf", type="f").dump(fig, formatter="mpl")

    # save metric as json
    json_metric = {"auc_score":auc_scores}
    json_path = "auc_score.json"
    output.child(json_path, type="f").dump(json_metric, formatter="json")




def plot_output_nodes(
        model: tf.keras.models.Model,
        train: DotDict,
        validation: DotDict,
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
) -> None:
    """
    Function that creates a plot for each ML output node,
    displaying all processes per plot.
    """
    # use CMS plotting style
    n_classes = len(train["target"][0])
    for i in range(n_classes):
        fig, ax = plt.subplots()

        var_title = f"Output node {process_insts[i].label}"

        h = (
            hist.Hist.new
            .StrCat(["train", "validation"], name="type")
            .IntCat([], name="process", growth=True)
            .Reg(20, 0, 1, name=var_title)
            .Weight()
        )

        for input_type, inputs in (("train", train), ("validation", validation)):
            for j in range(n_classes):
                inputs["target"] = tf.reshape(inputs["target"], (-1,n_classes))
                inputs["prediction"] = tf.reshape(inputs["prediction"], (-1,n_classes))


                mask = np.argmax(inputs["target"], axis=1) == j
                fill_kwargs = {
                    "type": input_type,
                    "process": j,
                    var_title: inputs["prediction"][:, i][mask],
                    "weight": inputs["weights"][mask],
                }
                h.fill(**fill_kwargs)
        plot_kwargs = {
            "ax": ax,
            "label": [proc_inst.label for proc_inst in process_insts],
            "color": ["red", "blue"],
        }

        # dummy legend entries
        plt.hist([], histtype="step", label="Training", color="black")
        plt.hist([], histtype="step", label="Validation (scaled)", linestyle="dotted", color="black")

        # plot training scores
        h[{"type": "train"}].plot1d(**plot_kwargs)

        # legend
        ax.legend(loc="best")

        ax.set(**{
            "ylabel": "Entries",
            "ylim": (0.00001, ax.get_ylim()[1]),
            "xlim": (0, 1),
        })

        # plot validation scores, scaled to train dataset
        scale = h[{"type": "train"}].sum().value / h[{"type": "validation"}].sum().value
        (h[{"type": "validation"}] * scale).plot1d(**plot_kwargs, linestyle="dotted")

        mplhep.cms.label(ax=ax, llabel="Work in progress", data=False, loc=0)
        output.child(f"Node_{process_insts[i].name}.pdf", type="f").dump(fig, formatter="mpl")

def plot_shap_values(
        model: tf.keras.models.Model,
        train: DotDict,
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
        target_dict,
        feature_names,
) -> None:
    feature_dict = {
    "mjj": r"$m_{jj}$",
    "mbjetbjet": r"$m_{bb}$",
    "mHH": r"$m_{HH}$",
    "mtautau": r"$m_{\tau\tau}$",
    "jet1_p": r"$jet1_p$",
    "jet2_p": r"$jet2_p$",
    "jet3_p": r"$jet3_p$",
    "jet4_p": r"$jet4_p$",
    "jet1_pt": r"$jet1_pt$",
    "jet2_pt": r"$jet2_pt$",
    "jet3_pt": r"$jet3_pt$",
    "jet4_pt": r"$jet4_pt$",
    "jet1_eta": r"$jet1_eta$",
    "jet2_eta": r"$jet2_eta$",
    "jet3_eta": r"$jet3_eta$",
    "jet4_eta": r"$jet4_eta$",
    "jet1_phi": r"$jet1_phi$",
    "jet2_phi": r"$jet2_phi$",
    "jet3_phi": r"$jet3_phi$",
    "jet4_phi": r"$jet4_phi$",
    "jet1_mass": r"$jet1_mass$",
    "jet2_mass": r"$jet2_mass$",
    "jet3_mass": r"$jet3_mass$",
    "jet4_mass": r"$jet4_mass$",
    "jet1_e": r"$jet1_e$",
    "jet2_e": r"$jet2_e$",
    "jet3_e": r"$jet3_e$",
    "jet4_e": r"$jet4_e$",
    "bjet1_p": r"$bjet1_p$",
    "bjet2_p": r"$bjet2_p$",
    "bjet3_p": r"$bjet3_p$",
    "bjet4_p": r"$bjet4_p$",
    "bjet1_pt": r"$bjet1_pt$",
    "bjet2_pt": r"$bjet2_pt$",
    "bjet3_pt": r"$bjet3_pt$",
    "bjet4_pt": r"$bjet4_pt$",
    "bjet1_eta": r"$bjet1_eta$",
    "bjet2_eta": r"$bjet2_eta$",
    "bjet3_eta": r"$bjet3_eta$",
    "bjet4_eta": r"$bjet4_eta$",
    "bjet1_phi": r"$bjet1_phi$",
    "bjet2_phi": r"$bjet2_phi$",
    "bjet3_phi": r"$bjet3_phi$",
    "bjet4_phi": r"$bjet4_phi$",
    "bjet1_mass": r"$bjet1_mass$",
    "bjet2_mass": r"$bjet2_mass$",
    "bjet3_mass": r"$bjet3_mass$",
    "bjet4_mass": r"$bjet4_mass$",
    "bjet1_e": r"$bjet1_e$",
    "bjet2_e": r"$bjet2_e$",
    "bjet3_e": r"$bjet3_e$",
    "bjet4_e": r"$bjet4_e$",
    "tau1_p": r"$tau1_p$",
    "tau2_p": r"$tau2_p$",
    "tau1_pt": r"$tau1_pt$",
    "tau2_pt": r"$tau2_pt$",
    "tau1_eta": r"$tau1_eta$",
    "tau2_eta": r"$tau2_eta$",
    "tau1_phi": r"$tau1_phi$",
    "tau2_phi": r"$tau2_phi$",
    "tau1_mass": r"$tau1_mass$",
    "tau2_mass": r"$tau2_mass$",
    "tau1_e": r"$tau1_e$",
    "tau2_e": r"$tau2_e$",
    "vbfjet1_pt": r"$vbfjet1_pt$",
    "vbfjet2_pt": r"$vbfjet2_pt$",
    "vbfjet1_eta": r"$vbfjet1_eta$",
    "vbfjet2_eta": r"$vbfjet2_eta$",
    "vbfjet1_phi": r"$vbfjet1_phi$",
    "vbfjet2_phi": r"$vbfjet2_phi$",
    "vbfjet1_mass": r"$vbfjet1_mass$",
    "vbfjet2_mass": r"$vbfjet2_mass$",
}

    # names of features and classes
    feature_list = [feature_dict[feature] for feature in feature_names[1]]
    feature_list.insert(0, 'Deep Sets')

    # make sure class names are sorted correctly in correspondence to their target index
    classes = sorted(target_dict.items(), key=lambda x: x[1])
    class_sorted = np.array(classes)[:, 0]
    class_list = ['empty' for i in range(len(process_insts))]
    for proc in process_insts:
        idx = np.where(class_sorted == proc.name)
        class_list[idx[0][0]] = proc.label

    # calculate shap values
    inp_deepSets = train['prediction_deepSets'].numpy()
    inp_ff = train['inputs2'].numpy()
    inp = np.concatenate((inp_deepSets, inp_ff), axis=1)
    explainer = shap.KernelExplainer(model, inp[:50])
    shap_values = explainer.shap_values(inp[-50:])

    # Feature Ranking
    fig1 = plt.figure()
    shap.summary_plot(shap_values, inp[:500], plot_type="bar",
        feature_names=feature_list, class_names=class_list)
    output.child("Feature_Ranking.pdf", type="f").dump(fig1, formatter="mpl")

    # Violin Plots
    for i, node in enumerate(class_list):
        fig2 = plt.figure()
        shap.summary_plot(shap_values[i], inp[:100], plot_type="violin",
            feature_names=feature_list, class_names=node)
        output.child(f"Violin_{class_sorted[i]}.pdf", type="f").dump(fig2, formatter="mpl")