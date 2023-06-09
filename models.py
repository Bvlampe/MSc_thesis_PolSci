import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, RocCurveDisplay, PrecisionRecallDisplay
from copy import deepcopy


def query_yn(question):
    answer = ""
    while answer not in ["y", "n"]:
        answer = input(question)
    return True if answer == "y" else False


# Allows for the creation of the main, as well as variant models
def new_models(varchoice, roclog, prlog, extra_options=[], write=True, debug_print=False, last_train_year=2009):
    assert(varchoice in ["academic", "professional", "combined", "all", "notrade", "nogdp", "nogdpweap"])
    main_data = pd.read_csv("merged_data.csv", index_col=[0, 1])

    if varchoice == "notrade":
        main_data.drop(["US Trade"], axis=1, inplace=True)
    elif varchoice == "nogdp":
        main_data.drop(["GDP"], axis=1, inplace=True)
    elif varchoice == "nogdpweap":
        main_data.drop(["GDP"], axis=1, inplace=True)
        main_data.drop(["Weapon imports"], axis=1, inplace=True)

    # Lag DV to avoid data leakage
    main_data["Terrorist attack lag-1"] = main_data.groupby(level=0)["Terrorist attack"].shift(-1)

    if "interpol_glob_GTD" in extra_options:
        main_data.replace(to_replace={"Global terrorist attacks": 1}, value=np.NaN, inplace=True)

    # Interpolate missing values for some sparsely-documented features,
    # max 10 consecutive years to be interpolated
    for col in main_data.columns:
        main_data[col] = main_data.groupby(level=0, group_keys=False)[col].apply(
            lambda group: group.interpolate(limit=10, limit_area="inside"))

    # Literacy and education datasets are used for the same reason, so I drop one. TBA: create models with either one
    # and compare performances
    if "education" in extra_options:
        main_data.drop(["Literacy"], axis=1, inplace=True)
    else:
        main_data.drop(["Education"], axis=1, inplace=True)
    main_data.dropna(inplace=True)

    # Ensures the "no data" value is the default case dropped in the dummies
    elecsys_dummies = pd.get_dummies(main_data["Elec_sys"], drop_first=False, prefix="elecsys").drop(["elecsys_No data"], axis=1)
    main_data = pd.concat([main_data, elecsys_dummies], axis=1).drop(["Elec_sys"], axis=1)
    main_data.loc[:, "Terrorist attack lag-1"].replace({True: 1, False: 0}, inplace=True)
    main_data.loc[:, "Intervention"].replace({True: 1, False: 0}, inplace=True)

    if varchoice == "academic":
        main_data.drop(["Terrorist attack", "US Trade", "Internet users", "Weapon imports",
                        "Global terrorist attacks"], axis=1, inplace=True)
    elif varchoice == "professional":
        main_data.drop(["Terrorist attack", "Fragility", "Durability", "Democracy", "FH_pol", "FH_civ", "GDP",
                        "Inequality", "Poverty", "Inflation", "Global terrorist attacks"],
                       axis=1, inplace=True)
        if "Literacy" in main_data.columns:
            main_data.drop(["Literacy"], axis=1, inplace=True)
        else:
            main_data.drop(["Education"], axis=1, inplace=True)
        main_data.drop(elecsys_dummies.columns, axis=1, inplace=True)
    elif varchoice == "combined":
        main_data.drop(["Terrorist attack", "Global terrorist attacks"], axis=1, inplace=True)

    model_logreg = LogisticRegression(max_iter=1000)
    model_rf = RandomForestClassifier()
    model_gbm = GradientBoostingClassifier()

    indep_vars = list(main_data.columns.values)
    indep_vars.remove("Terrorist attack lag-1")
    log = pd.DataFrame(index=indep_vars, columns=["LR", "RF", "RF importance scaled", "GBM", "GBM importance scaled"])

    # Split off data (by default, cases after 2007, or about 25% of the data) for testing
    main_data['Year'] = main_data.index.get_level_values(1)
    data_rem = main_data.loc[main_data['Year'].isin(range(last_train_year + 1))]
    data_test = main_data.loc[main_data['Year'].isin(range(last_train_year + 1, 9999))]
    y_test = data_test['Terrorist attack lag-1']
    x_test = data_test.drop(["Terrorist attack lag-1", "Year"], axis=1)

    if debug_print:
        print(f"Model variation: {varchoice}, extra_options={extra_options}:")
        print(f"Length of train/val set: {len(data_rem)}")
        print(f"Length of test set: {len(data_test)}")
        print(f"Split: {len(data_rem) / (len(data_rem) + len(data_test))}/{len(data_test) / (len(data_rem) + len(data_test))}")
        print()

    best_model_lr = None
    best_prauc_lr = 0
    best_model_rf = None
    best_prauc_rf = 0
    best_model_gbm = None
    best_prauc_gbm = 0

    n_splits = 5
    tss = TimeSeriesSplit(n_splits=n_splits)

    for train_index, test_index in tss.split(data_rem['Year'].unique()):
        x_train, x_val = data_rem.loc[data_rem['Year'].isin(data_rem['Year'].unique()[train_index])], data_rem.loc[
            data_rem['Year'].isin(data_rem['Year'].unique()[test_index])]
        # I have no idea what the next two lines are meant to do but I'd rather keep them in lest I break something
        x_train.columns = x_train.columns.tolist()
        x_val.columns = x_val.columns.tolist()

        y_train, y_val = x_train['Terrorist attack lag-1'], x_val['Terrorist attack lag-1']
        x_train = x_train.drop(["Terrorist attack lag-1", "Year"], axis=1)
        x_val = x_val.drop(["Terrorist attack lag-1", "Year"], axis=1)

        model_logreg.fit(x_train, y_train)
        precision, recall, thresholds = precision_recall_curve(y_val, model_logreg.predict_proba(x_val)[:, 1])
        if auc(recall, precision) > best_prauc_lr:
            best_prauc_lr = auc(recall, precision)
            best_model_lr = deepcopy(model_logreg)

        model_rf.fit(x_train, y_train)
        precision, recall, thresholds = precision_recall_curve(y_val, model_rf.predict_proba(x_val)[:, 1])
        if auc(recall, precision) > best_prauc_rf:
            best_prauc_rf = auc(recall, precision)
            best_model_rf = deepcopy(model_rf)

        model_gbm.fit(x_train, y_train)
        precision, recall, thresholds = precision_recall_curve(y_val, model_gbm.predict_proba(x_val)[:, 1])
        if auc(recall, precision) > best_prauc_gbm:
            best_prauc_gbm = auc(recall, precision)
            best_model_gbm = deepcopy(model_gbm)

    y_pred = best_model_lr.predict(x_test)
    log.loc["Accuracy", "LR"] = accuracy_score(y_test, y_pred)
    log.loc["Precision", "LR"] = precision_score(y_test, y_pred)
    log.loc["Recall", "LR"] = recall_score(y_test, y_pred)
    log.loc["ROC-AUC", "LR"] = roc_auc_score(y_test, best_model_lr.predict_proba(x_test)[:,1])
    precision, recall, thresholds = precision_recall_curve(y_test, best_model_lr.predict_proba(x_test)[:, 1])
    log.loc["PR-AUC", "LR"] = auc(recall, precision)

    y_pred = best_model_rf.predict(x_test)
    log.loc[:, "RF"].iloc[:len(best_model_rf.feature_importances_)] = best_model_rf.feature_importances_
    log.loc[:, "RF importance scaled"].iloc[:len(best_model_rf.feature_importances_)] = \
        [imp / max(best_model_rf.feature_importances_) for imp in best_model_rf.feature_importances_]
    log.loc["Accuracy", "RF"] = accuracy_score(y_test, y_pred)
    log.loc["Precision", "RF"] = precision_score(y_test, y_pred)
    log.loc["Recall", "RF"] = recall_score(y_test, y_pred)
    log.loc["ROC-AUC", "RF"] = roc_auc_score(y_test, model_rf.predict_proba(x_test)[:,1])
    precision, recall, thresholds = precision_recall_curve(y_test, best_model_rf.predict_proba(x_test)[:, 1])
    log.loc["PR-AUC", "RF"] = auc(recall, precision)

    y_pred = model_gbm.predict(x_test)
    log.loc[:, "GBM"].iloc[:len(best_model_gbm.feature_importances_)] = best_model_gbm.feature_importances_
    log.loc[:, "GBM importance scaled"].iloc[:len(best_model_rf.feature_importances_)] = \
        [imp / max(best_model_gbm.feature_importances_) for imp in best_model_gbm.feature_importances_]
    log.loc["Accuracy", "GBM"] = accuracy_score(y_test, y_pred)
    log.loc["Precision", "GBM"] = precision_score(y_test, y_pred)
    log.loc["Recall", "GBM"] = recall_score(y_test, y_pred)
    log.loc["ROC-AUC", "GBM"] = roc_auc_score(y_test, best_model_gbm.predict_proba(x_test)[:, 1])
    precision, recall, thresholds = precision_recall_curve(y_test, best_model_gbm.predict_proba(x_test)[:, 1])
    log.loc["PR-AUC", "GBM"] = auc(recall, precision)

    suffix = '_' + "+".join(extra_options) if extra_options else ''
    if write:
        log.to_csv("output_files/" + varchoice + suffix + ".csv")

    for model in ["LR", "RF", "GBM"]:
        roclog.loc[varchoice, model] = log.loc["ROC-AUC", model]
        prlog.loc[varchoice, model] = log.loc["PR-AUC", model]

    # Plot ROC and PR curves for the main model
    if not extra_options and varchoice == "all":
        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        RocCurveDisplay.from_estimator(best_model_lr, x_test, y_test).plot(ax=ax[0][0])
        ax[0][0].title.set_text("ROC curve, LR")
        plt.close()
        RocCurveDisplay.from_estimator(best_model_rf, x_test, y_test).plot(ax=ax[0][1])
        ax[0][1].title.set_text("ROC curve, RF")
        plt.close()
        RocCurveDisplay.from_estimator(best_model_gbm, x_test, y_test).plot(ax=ax[0][2])
        ax[0][2].title.set_text("ROC curve, GBM")
        plt.close()
        PrecisionRecallDisplay.from_estimator(best_model_lr, x_test, y_test).plot(ax=ax[1][0])
        ax[1][0].title.set_text("PR curve, LR")
        plt.close()
        PrecisionRecallDisplay.from_estimator(best_model_rf, x_test, y_test).plot(ax=ax[1][1])
        ax[1][1].title.set_text("PR curve, RF")
        plt.close()
        PrecisionRecallDisplay.from_estimator(best_model_gbm, x_test, y_test).plot(ax=ax[1][2])
        ax[1][2].title.set_text("PR curve, GBM")
        plt.close()
        fig.tight_layout(h_pad=3.0)
        fig.savefig("plots/curves.png")
        plt.close()
