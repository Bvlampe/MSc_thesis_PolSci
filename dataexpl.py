import pandas as pd
import matplotlib.pyplot as plt

def dataexpl():
    main_data = pd.read_csv("merged_data.csv", index_col=[0, 1])
    print("Total length:", len(main_data))
    print("Units with terrorist attacks:", len(main_data[main_data["Terrorist attack"] == True]))
    perc = 100 * len(main_data[main_data["Terrorist attack"] == True]) / len(main_data)
    print("Percentage of units with terrorist attacks:", perc, "%")

    main_data.loc["Afghanistan", "Global terrorist attacks"].plot(title="Global terrorist attacks")
    plt.show()