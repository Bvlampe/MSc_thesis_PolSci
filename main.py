import pandas as pd
import dataexpl
import dataprep
import models
import resexpl


def main():
    dataprep.dataprep("edit", edit_col="Global terrorism")
    dataexpl.dataexpl()
    for extra in [[], ["interpol_glob_GTD"], ["education"], ["interpol_glob_GTD", "education"]]:
        auc_roc_log = pd.DataFrame()
        auc_pr_log = pd.DataFrame()
        for var in ["academic", "professional", "combined", "all", "notrade", "nogdp"]:
            models.new_models(varchoice=var, roclog=auc_roc_log, prlog=auc_pr_log, extra_options=extra, write=True)
        suffix = '_' + "+".join(extra) if extra else ''
        auc_roc_log.to_csv("output_files/auc_roc_per_model" + suffix + ".csv")
        auc_pr_log.to_csv("output_files/auc_pr_per_model" + suffix + ".csv")
    auc_roc_log = pd.DataFrame()
    auc_pr_log = pd.DataFrame()
    models.new_models(varchoice="nogdpweap", roclog=auc_roc_log, prlog=auc_pr_log, write=True)
    resexpl.resexpl()


if __name__ == "__main__":
    main()
