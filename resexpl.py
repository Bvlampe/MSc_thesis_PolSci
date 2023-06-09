import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



def resexpl():
    path = "output_files/"
    acad_data = pd.read_csv(path + "academic.csv", index_col=0)
    prof_data = pd.read_csv(path + "professional.csv", index_col=0)
    comb_data = pd.read_csv(path + "combined.csv", index_col=0)
    all_data = pd.read_csv(path + "all.csv", index_col=0)

    i = 1
    fig, ax = plt.subplots(2, 2)
    mpl.rc('xtick', labelsize=8)
    mpl.rc('ytick', labelsize=8)
    for (set, name) in [(acad_data, "Academic"), (prof_data, "Professional"), (comb_data, "Combined"), (all_data, "All")]:
        data = set.transpose()
        plt.subplot(2, 2, i)
        x = set.loc["PR-AUC", :].dropna()
        y = set.loc["ROC-AUC", :].dropna()
        labels = set.columns
        ax[0 if i < 3 else 1][(i + 1) % 2].scatter(x, y)
        plt.xlabel("AUC-PR", fontsize=8)
        plt.ylabel("AUC-ROC", fontsize=8)
        plt.xticks(np.arange(0.84, 0.91, 0.02))
        plt.yticks(np.arange(0.78, 0.89, 0.02))
        for label in labels:
            point = (set.loc["PR-AUC", label], set.loc["ROC-AUC", label])
            coords = (set.loc["PR-AUC", label] + 0.003, set.loc["ROC-AUC", label])
            ax[0 if i < 3 else 1][(i + 1) % 2].annotate(label, point, coords, xycoords="data")
        plt.title(name)
        i += 1
    fig.tight_layout()
    plt.savefig("plots/AUC-ROC over AUC-PR.png")
    plt.show()
