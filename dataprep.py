import pandas as pd


# Cuts unnecessary data from the 100MB GTD and exports the 10x smaller dataframe back to a .csv file
def cut_GTD(path_in, path_out, code="cp1252"):
    in_data = pd.read_csv(path_in, encoding=code)
    in_data = in_data[["eventid", "iyear", "country", "country_txt", "success", "INT_LOG", "INT_IDEO"]]
    in_data.to_csv(path_out)
    return in_data


# Transforms the GTD into a country-year format
def format_GTD(in_data):
    out_data = pd.DataFrame(columns=range(in_data["iyear"].min(), in_data["iyear"].max() + 1))
    for _, row in in_data.iterrows():
        if True: # Add possible inclusion criteria here
            out_data.loc[row["country_txt"], row["iyear"]] = True
    out_data.fillna(value=False, inplace=True)
    return out_data


# Transforms the election system dataset from an election-level format to a country-year format
def format_elecsys(in_data):
    return False


def dataprep():
    path_rawdata = "datasets_input/"

    path_GTD_raw = path_rawdata + "GTD_raw.csv"
    path_GTD = path_rawdata + "GTD.csv"
    path_fragility = path_rawdata + "fragility.csv"
    # [Regime durability]
    path_elecsys = path_rawdata + "electoral_system.csv"
    path_glob = path_rawdata + "globalisation.csv"
    path_iusers = path_rawdata + "internet_users.csv"
    path_lit = path_rawdata + "literacy_rate.csv"

    # raw_GTD = cut_GTD(path_GTD_raw, path_GTD)
    raw_GTD = pd.read_csv(path_GTD, encoding="cp1252")
    raw_fragility = pd.read_csv(path_fragility, sep=';')
    # [Regime durability]
    raw_elecsys = pd.read_csv(path_elecsys)
    raw_glob = pd.read_csv(path_glob, encoding="cp1252")
    raw_iusers = pd.read_csv(path_iusers)
    raw_lit = pd.read_csv(path_lit)
    print(raw_elecsys.head())