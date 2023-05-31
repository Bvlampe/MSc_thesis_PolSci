import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def dataexpl():
    main_data = pd.read_csv("merged_data.csv", index_col=[0, 1])
    print("Total length:", len(main_data))
    print("Units with terrorist attacks:", len(main_data[main_data["Terrorist attack"] == True]))
    perc = 100 * len(main_data[main_data["Terrorist attack"] == True]) / len(main_data)
    print("Percentage of units with terrorist attacks:", perc, "%")

    main_data.loc["Afghanistan", "Global terrorist attacks"].plot(title="Global terrorist attacks")
    plt.show()

    attacks = pd.DataFrame(columns=["Population", "Attacks"])
    for ctry, ctry_data in main_data.groupby(axis=0, level=0):
        attacks.loc[ctry, "Population"] = ctry_data.iloc[-1, :].loc["Population"]
        attacks.loc[ctry, "Attacks"] =\
            len(ctry_data[ctry_data["Terrorist attack"] == True]) / ctry_data.loc[:, "Terrorist attack"].count()
    attacks.dropna(inplace=True)
    attacks.sort_values(by="Population", inplace=True)
    attacks.plot(x="Population", y="Attacks", kind="scatter", logx=True, title="Terrorist attack frequency over population")
    plt.show()