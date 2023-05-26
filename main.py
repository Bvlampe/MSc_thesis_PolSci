import pandas as pd
import dataexpl
import dataprep
import models


def main():
    # dataprep.dataprep("edit", edit_col="Global terrorism")
    # dataexpl.dataexpl()
    # for extra in [[], ["interpol_glob_GTD"], ["education"], ["interpol_glob_GTD", "education"]]:
    #     auc_roc_set = pd.DataFrame()
    #     for vars in ["academic", "professional", "combined", "all", "notrade"]:
    #         models.partial_models(varchoice=vars, macrolog=auc_roc_set, extra_options=extra)
    #     suffix = '_' + "+".join(extra) if extra else ''
    #     auc_roc_set.to_csv("output_files/auc_roc_per_model" + suffix + ".csv")
    auc_roc_set = pd.DataFrame()
    models.partial_models(varchoice="nogdp", macrolog=auc_roc_set, extra_options=[])
    auc_roc_set.to_csv("output_files/auc_roc_per_model_nogdp.csv")
    # models.models()


if __name__ == "__main__":
    main()
