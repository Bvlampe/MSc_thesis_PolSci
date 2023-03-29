import pandas as pd


# Cuts unnecessary data from the 100MB GTD and exports the 10x smaller dataframe back to a .csv file
def cut_GTD(path_in, path_out, code="cp1252"):
    in_data = pd.read_csv(path_in, encoding=code)
    in_data = in_data[["eventid", "iyear", "country", "country_txt", "success", "INT_LOG", "INT_IDEO"]]
    in_data.to_csv(path_out)
    return in_data


# Takes the country names in the main dataset (as list-like variable) and a DataFrame of country names in
# the other datasets (each column being a list of all countries from one specific dataset, with the column
# name being the name of the dataset) and exports an Excel file with
def create_concordance_table(main, in_data):
    first = list(set(main))
    first.sort()
    out = pd.DataFrame(index=range(len(list(first))), columns=["Main"], data=list(first))
    for column in in_data.columns:
        second = set(in_data.loc[:, column])
        both = set(first).intersection(second)
        unique = list(second - both)

        i = 0
        for name in unique:
            out.loc[i, column] = name
            i += 1
    out.to_csv("country_names.csv", index=False)


# Transforms the GTD into a country-year format
def format_GTD(in_data):
    out_data = pd.DataFrame(columns=range(in_data["iyear"].min(), in_data["iyear"].max() + 1))
    for _, row in in_data.iterrows():
        if True: # Add possible inclusion criteria here
            out_data.loc[row["country_txt"], row["iyear"]] = True
    out_data.fillna(value=False, inplace=True)
    return out_data


# Transforms the election system dataset from an election-level format to a country-year format
# WIP: main dataset structure update in progress, then come back here
def format_elecsys(in_elec, in_main_structure, inplace=False):
    out_data = pd.DataFrame(index=in_main_structure.index, columns=["Elec_sys"])
    print(in_elec.columns)
    for _, row in in_elec.iterrows():
        out_data.loc[(row["Country"], row["Year"]), "Elec_sys"] = row["Electoral system family"]

    # Gotta do this by country otherwise a recent election might propagate to the early years of the next country
    for country in out_data.index.get_level_values(0):
        out_data.loc[country, :].fillna(method="ffill", inplace=True)
    # out_data.fillna(method="ffill", inplace=True)
    if inplace:
        in_elec = out_data
    else:
        return out_data



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

    main_index_ctry = raw_GTD.loc[:, "country_txt"].unique()
    main_index_year = range(raw_GTD.loc[:, "iyear"].min(), raw_GTD.loc[:, "iyear"].max() + 1)
    main_index = pd.MultiIndex.from_product([main_index_ctry, main_index_year], names=["Country", "Year"])
    main_data = pd.DataFrame(index=main_index)

    cntry_names = pd.DataFrame()
    cntry_names["Fragility"] = raw_fragility.loc[:, "country"].unique()
    # [Regime durability]
    i = 0
    for name in raw_elecsys.loc[:, "Country"].unique():
        cntry_names.loc[i, "Election system"] = name
        i += 1
    create_concordance_table(main_data.index.get_level_values(0), cntry_names)

    # print(format_elecsys(raw_elecsys, main_data))
