import dataexpl
import dataprep
import models


def main():
    # dataprep.dataprep("edit", edit_col="Global terrorism")
    # dataexpl.dataexpl()
    for vars in ["academic", "professional", "combined", "all"]:
        models.partial_models(varchoice=vars, extra_options=[])
    # models.models()


if __name__ == "__main__":
    main()
