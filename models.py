import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve


def query_yn(question):
    answer = ""
    while answer not in ["y", "n"]:
        answer = input(question)
    return True if answer == "y" else False

def models():
    main_data = pd.read_csv("merged_data.csv", index_col=[0, 1])

    # Lag DV to avoid data leakage
    main_data["Terrorist attack lag-1"] = main_data.groupby(level=0)["Terrorist attack"].shift(-1)
    # Drop un-lagged DV and variables with too little data, as well as missing values
    main_data.drop(["Terrorist attack"], axis=1, inplace=True)

    nans_before = main_data.isna().sum()
    print("NaN per column before interpolation:\n", nans_before, end='\n')

    # Interpolate missing values for some sparsely-documented features,
    # max 10 consecutive years to be interpolated
    for col in main_data.columns:
        main_data[col] = main_data.groupby(level=0)[col].apply(
            lambda group: group.interpolate(limit=10, limit_area="inside"))

    print("NaN per column after interpolation:\n", main_data.isna().sum(), end='\n')
    print("NaN interpolated per column:\n", nans_before - main_data.isna().sum(), end='\n')

    # Literacy and education datasets are used for the same reason so I drop one. TBA: create models with either one
    # and compare performances
    print("main_data shape before dropping:", main_data.shape, end='\n')
    main_data.drop(["Education"], axis=1, inplace=True)
    main_data.dropna(inplace=True)
    print("main_data shape after dropping:", main_data.shape, end='\n')

    if not query_yn("Continue with model generation? y/n: "):
        return

    # Ensures the "no data" value is the default case dropped in the dummies
    elecsys_dummies = pd.get_dummies(main_data["Elec_sys"], drop_first=False, prefix="elecsys").drop(["elecsys_No data"], axis=1)
    main_data = pd.concat([main_data, elecsys_dummies], axis=1).drop(["Elec_sys"], axis=1)
    main_data.loc[:, "Terrorist attack lag-1"].replace({True: 1, False: 0}, inplace=True)
    main_data.loc[:, "Intervention"].replace({True: 1, False: 0}, inplace=True)

    main_data['Year'] = main_data.index.get_level_values(1)
    n_splits = 5


    tss = TimeSeriesSplit(n_splits=n_splits)
    model_logreg = LogisticRegression(max_iter=1000)
    model_rf = RandomForestClassifier()
    model_gbm = GradientBoostingClassifier()

    indep_vars = list(main_data.columns.values)
    indep_vars.remove("Terrorist attack lag-1")
    # indep_vars.remove("Year")


    log = pd.DataFrame(index=indep_vars)

    i = 1
    for train_index, test_index in tss.split(main_data['Year'].unique()):
        print("Fold number", i, ":")
        # Get the training and testing data for this split
        # x_train, x_test = main_data.iloc[train_index][indep_vars], main_data.iloc[test_index][indep_vars]
        # y_train, y_test = main_data.iloc[train_index]["Terrorist attack"], main_data.iloc[test_index]["Terrorist attack"]

        x_train, x_test = main_data.loc[main_data['Year'].isin(main_data['Year'].unique()[train_index])], main_data.loc[
            main_data['Year'].isin(main_data['Year'].unique()[test_index])]
        x_train.columns = x_train.columns.tolist()
        x_test.columns = x_test.columns.tolist()

        y_train, y_test = x_train['Terrorist attack lag-1'], x_test['Terrorist attack lag-1']
        x_train = x_train.drop(["Terrorist attack lag-1"], axis=1)
        x_test = x_test.drop(["Terrorist attack lag-1"], axis=1)


        model_logreg.fit(x_train, y_train)
        y_pred = model_logreg.predict(x_test)
        print("Logistic regression:")
        # print(classification_report(y_test, y_pred))
        # print(confusion_matrix(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred), "Recall:", recall_score(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("ROC-AUC-score: ", roc_auc_score(y_test, model_logreg.predict_proba(x_test)[:,1]))
        print()

        model_rf.fit(x_train, y_train)
        y_pred = model_rf.predict(x_test)
        print("Random forest:")
        print("Precision:", precision_score(y_test, y_pred), "Recall:", recall_score(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("ROC-AUC-score: ", roc_auc_score(y_test, model_rf.predict_proba(x_test)[:, 1]))
        print()

        model_gbm.fit(x_train, y_train)
        y_pred = model_gbm.predict(x_test)
        print("Gradient boosting machine:")
        print("Precision:", precision_score(y_test, y_pred), "Recall:", recall_score(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("ROC-AUC-score: ", roc_auc_score(y_test, model_gbm.predict_proba(x_test)[:,1]))


        # The feature_importances_ attribute measures the "mean and standard deviation of accumulation
        # of the impurity decrease within each tree" - scikit-learn doc
        for feature, v in zip(x_train.columns, model_gbm.feature_importances_):
            print(f"Feature: {feature}, Score: %.5f" % (v))

        log.loc[:len(model_gbm.feature_importances_),f"Fold {i}"] = model_gbm.feature_importances_
        log.loc["Accuracy", f"Fold {i}"] = accuracy_score(y_test, y_pred)
        log.loc["Precision", f"Fold {i}"] = precision_score(y_test, y_pred)
        log.loc["Recall", f"Fold {i}"] = recall_score(y_test, y_pred)
        log.loc["ROC-AUC", f"Fold {i}"] = roc_auc_score(y_test, model_gbm.predict_proba(x_test)[:,1])
        print("--------------------------------------------------------")
        i += 1
    print("Log file:\n", log)
    if query_yn("Write model output to log file? y/n: "):
        log.to_csv("log.csv")
