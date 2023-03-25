import pandas as pd


def dataprep():
    path_rawdata = "datasets_input/"

    path_glob = path_rawdata + "globalisation.csv"
    path_iusers = path_rawdata + "internet_users.csv"
    path_lit = path_rawdata + "literacy_rate.csv"

    raw_glob = pd.read_csv(path_glob, encoding='cp1252')
    raw_iusers = pd.read_csv(path_iusers)
    raw_lit = pd.read_csv(path_lit)
    print(raw_glob.head())