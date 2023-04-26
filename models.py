import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



def models():
    main_data = pd.read_csv("merged_data.csv", index_col=[0, 1])
    main_data.drop(["Inequality", "Poverty", "Literacy"], axis=1, inplace=True)
    main_data.dropna(inplace=True)
    # Ensures the "no data" value is the default case dropped in the dummies
    elecsys_dummies = pd.get_dummies(main_data["Elec_sys"], drop_first=False, prefix="elecsys").drop(["elecsys_No data"], axis=1)
    main_data = pd.concat([main_data, elecsys_dummies], axis=1).drop(["Elec_sys"], axis=1)
    main_data.loc[:, "Terrorist attack"].replace({True: 1, False: 0}, inplace=True)
    main_data.loc[:, "Intervention"].replace({True: 1, False: 0}, inplace=True)

    main_data['Year'] = main_data.index.get_level_values(1)
    n_splits = 4


    tss = TimeSeriesSplit(n_splits=n_splits)
    model = LogisticRegression(max_iter=500)

    indep_vars = list(main_data.columns.values)
    indep_vars.remove("Terrorist attack")
    indep_vars.remove("Year")

    for train_index, test_index in tss.split(main_data['Year'].unique()):
        # Get the training and testing data for this split
        # x_train, x_test = main_data.iloc[train_index][indep_vars], main_data.iloc[test_index][indep_vars]
        # y_train, y_test = main_data.iloc[train_index]["Terrorist attack"], main_data.iloc[test_index]["Terrorist attack"]

        x_train, x_test = main_data.loc[main_data['Year'].isin(main_data['Year'].unique()[train_index])], main_data.loc[
            main_data['Year'].isin(main_data['Year'].unique()[test_index])]
        x_train.columns = x_train.columns.tolist()
        x_test.columns = x_test.columns.tolist()

        y_train, y_test = x_train['Terrorist attack'], x_test['Terrorist attack']
        x_train.drop(["Terrorist attack"], axis=1, inplace=True)
        x_test.drop(["Terrorist attack"], axis=1, inplace=True)

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        print(classification_report(y_test, predictions))
        print(confusion_matrix(y_test, predictions))
