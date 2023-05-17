import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve


def models():
    main_data = pd.read_csv("merged_data.csv", index_col=[0, 1])

    # Lag DV to avoid data leakage
    main_data["Terrorist attack lag-1"] = main_data.groupby(level=0)["Terrorist attack"].shift(-1)

    # Drop un-lagged DV and variables with too little data, as well as missing values
    main_data.drop(["Terrorist attack"], axis=1, inplace=True)
    main_data.drop(["Inequality", "Poverty", "Literacy", "Education"], axis=1, inplace=True)
    main_data.dropna(inplace=True)
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
    indep_vars.remove("Year")

    for train_index, test_index in tss.split(main_data['Year'].unique()):
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
        predictions = model_logreg.predict(x_test)
        print("Logistic regression:")
        # print(classification_report(y_test, predictions))
        # print(confusion_matrix(y_test, predictions))
        print("ROC-AUC-score: ", roc_auc_score(y_test, model_logreg.predict_proba(x_test)[:,1]))

        model_rf.fit(x_train, y_train)
        y_pred = model_rf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Random forest:")
        # print("Accuracy:", accuracy)
        print("ROC-AUC-score: ", roc_auc_score(y_test, model_rf.predict_proba(x_test)[:, 1]))

        model_gbm.fit(x_train, y_train)
        y_pred = model_gbm.predict(x_test)
        print("Gradient boosting machine:")
        print("ROC-AUC-score: ", roc_auc_score(y_test, model_gbm.predict_proba(x_test)[:,1]))


        # The feature_importances_ attribute measures the "mean and standard deviation of accumulation
        # of the impurity decrease within each tree" - scikit-learn doc
        importance = model_gbm.feature_importances_
        features = model_gbm.feature_names_in_
        # summarize feature importance
        for i, v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i, v))

        print("--------------------------------------------------------")
