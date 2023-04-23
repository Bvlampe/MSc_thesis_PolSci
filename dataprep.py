import pandas as pd
import math


# Cuts unnecessary data from the 100MB GTD and exports the 10x smaller dataframe back to a .csv file
def cut_GTD(path_in, path_out, code="cp1252"):
    in_data = pd.read_csv(path_in, encoding=code)
    in_data = in_data[["eventid", "iyear", "country", "country_txt", "success", "INT_LOG", "INT_IDEO"]]
    in_data.to_csv(path_out)
    return in_data


def list_countries_per_set(in_dataset, name_column, name_dataset, io_list):
    i = 0
    for country in in_dataset.loc[:, name_column].unique():
        io_list.loc[i, name_dataset] = country
        i += 1


# Takes the country names in the main dataset (as list-like variable) and a DataFrame of country names in
# the other datasets (each column being a list of all countries from one specific dataset, with the column
# name being the name of the dataset) and exports an Excel file with those countries not matching up between the
# datasets with the goal to create a concordance table from it
def create_country_table(main, in_data):
    first = list(set(main))
    first.sort()
    out = pd.DataFrame(index=range(len(list(first))), columns=["Main"], data=list(first))
    for column in in_data.columns:
        second = set(in_data.loc[:, column])
        both = set(first).intersection(second)
        # Necessary for some reason because a nan value managed to slip through in the Fragility dataset
        unique = [x for x in list(second - both) if type(x) == str]
        unique.sort()

        i = 0
        for name in unique:
            out.loc[i, column] = name
            i += 1
    out.to_csv("country_names.csv", index=False)


# Transforms the GTD into a country-year format
def format_GTD(in_data, in_index):
    out_data = pd.DataFrame(index=in_index)
    for _, row in in_data.iterrows():
        if True: # Add possible inclusion criteria here
            out_data.loc[row["country_txt"], row["iyear"]] = True
            out_data.loc[(row["country_txt"], row["iyear"]), "Terrorist attack"] = True
    out_data.fillna(value=False, inplace=True)
    # out_data.sort_index(inplace=True)
    return out_data


# Transforms the election system dataset from an election-level format to a country-year format
# WIP: main dataset structure update in progress, then come back here
def format_elecsys(in_elec, in_main_structure, inplace=False):
    out_data = pd.DataFrame(index=in_main_structure, columns=["Elec_sys"])
    for _, row in in_elec.iterrows():
        out_data.loc[(row["Country"], row["Year"]), "Elec_sys"] = row["Electoral system family"]
    out_data = out_data.groupby("Country").transform(lambda group: group.ffill())

    if inplace:
        in_elec = out_data
    else:
        return out_data


def calc_rel_frag(in_reldata, in_index):
    out_data = pd.DataFrame(index=in_index, columns=["Religious fragmentation"])
    for _, row in in_reldata.iterrows():
        # Formula here
        x = 0
    return out_data


def dataprep():
    path_rawdata = "datasets_input/"

    path_GTD_raw = path_rawdata + "GTD_raw.csv"
    path_GTD = path_rawdata + "GTD.csv"
    path_fragility = path_rawdata + "fragility.csv"
    # [Regime durability]
    path_elecsys = path_rawdata + "electoral_system.csv"
    # [Level of democracy]
    # [Political rights]
    # [Civil rights]
    path_inequality = path_rawdata + "inequality.csv"
    path_poverty = path_rawdata + "poverty.csv"
    # [Inflation]
    path_lit = path_rawdata + "literacy_rate.csv"
    path_iusers = path_rawdata + "internet_users.csv"
    path_interventions = path_rawdata + "interventions.csv"
    path_religion = path_rawdata + "religion.csv"
    path_glob = path_rawdata + "globalisation.csv"

    # raw_GTD = cut_GTD(path_GTD_raw, path_GTD)
    raw_GTD = pd.read_csv(path_GTD, encoding="cp1252")
    raw_fragility = pd.read_csv(path_fragility, sep=';')
    # [Regime durability]
    raw_elecsys = pd.read_csv(path_elecsys)
    # [Level of democracy]
    # [Political rights]
    # [Civil rights]
    raw_inequality = pd.read_csv(path_inequality)
    raw_poverty = pd.read_csv(path_poverty)
    # [Inflation]
    raw_lit = pd.read_csv(path_lit)
    raw_iusers = pd.read_csv(path_iusers)
    # raw_interventions = pd.read_csv(path_interventions)
    raw_religion = pd.read_csv(path_religion)
    raw_glob = pd.read_csv(path_glob, encoding="cp1252")

    main_index_ctry = raw_GTD.loc[:, "country_txt"].unique()
    main_index_ctry.sort()
    main_index_year = range(raw_GTD.loc[:, "iyear"].min(), raw_GTD.loc[:, "iyear"].max() + 1)
    main_index = pd.MultiIndex.from_product([main_index_ctry, main_index_year], names=["Country", "Year"])
    main_data = pd.DataFrame(index=main_index)
    main_data = format_GTD(raw_GTD, main_index)
    main_data.to_csv(path_rawdata + "GTD_formatted.csv")
    format_elecsys(raw_elecsys, main_index, inplace=True)

    cntry_names = pd.DataFrame()
    cntry_names["Fragility"] = raw_fragility.loc[:, "country"].unique()
    # [Regime durability]
    list_countries_per_set(raw_elecsys, "Country", "Election system", cntry_names)
    # [Level of democracy]
    # [Political rights]
    # [Civil rights]
    list_countries_per_set(raw_inequality, "Country Name", "Inequality", cntry_names)
    list_countries_per_set(raw_poverty, "Country Name", "Poverty", cntry_names)
    # [Inflation]
    list_countries_per_set(raw_lit, "Entity", "Literacy rate", cntry_names)
    list_countries_per_set(raw_iusers, "Country Name", "Internet users", cntry_names)
    # [Foreign interventions]
    list_countries_per_set(raw_religion, "Country", "Religion", cntry_names)
    list_countries_per_set(raw_glob, "country", "Globalisation", cntry_names)

    create_country_table(main_data.index.get_level_values(0), cntry_names)

