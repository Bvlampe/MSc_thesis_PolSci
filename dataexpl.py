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
    main_data.replace(to_replace={"Global terrorist attacks": 1}, value=np.NaN).loc["Afghanistan", "Global terrorist attacks"].interpolate().plot(title="Global terrorist attacks")
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

    main_data['Year'] = main_data.index.get_level_values(1)
    for col in main_data.columns:
        main_data[col] = main_data.groupby(level=0)[col].apply(
            lambda group: group.interpolate(limit=10, limit_area="inside"))
    main_data.dropna(inplace=True)
    for year in main_data.index.get_level_values(1).unique().sort_values():
        uptoincl = len(main_data.loc[main_data['Year'].isin(range(year + 1))])
        print(f"Percentage of cases up to and including year {year}: {100 * uptoincl / len(main_data)}%")
