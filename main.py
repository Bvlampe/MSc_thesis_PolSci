import pandas as pd
import dataexpl
import dataprep
import models


def main():
    # dataprep.dataprep("edit", edit_col="Global terrorism")
    # dataexpl.dataexpl()
    auc_roc_set = pd.DataFrame()
    for extra in [[], ["interpol_glob_GTD"], ["education"], ["interpol_glob_GTD", "education"]]:
        for vars in ["academic", "professional", "combined", "all"]:
            models.partial_models(varchoice=vars, macrolog=auc_roc_set, extra_options=extra)
        suffix = '_' + "+".join(extra) if extra else ''
        auc_roc_set.to_csv("output_files/auc_roc_per_model" + suffix + ".csv")
    # models.models()


if __name__ == "__main__":
    main()
