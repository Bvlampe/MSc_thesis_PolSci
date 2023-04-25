import pandas as pd
import math


# Cuts unnecessary data from the 100MB GTD and exports the 10x smaller dataframe back to a .csv file
def cut_GTD(path_in, path_out, code="cp1252"):
    in_data = pd.read_csv(path_in, encoding=code)
    in_data = in_data[["eventid", "iyear", "country", "country_txt", "success", "INT_LOG", "INT_IDEO"]]
    in_data.to_csv(path_out)
    return in_data


def list_countries_per_set(in_dataset, name_dataset, io_list, name_column="Country", in_index=False):
    i = 0
    countries = in_dataset.index.unique() if in_index else in_dataset.loc[:, name_column].unique()
    for country in countries:
        io_list.loc[i, name_dataset] = country
        i += 1


# Takes the country names in the main dataset (as list-like variable) and a DataFrame of country names in
# the other datasets (each column being a list of all countries from one specific dataset, with the column
# name being the name of the dataset) and exports an Excel file with those countries not matching up between the
# datasets with the goal to create a concordance table from it
def create_country_table(main, in_data, write=False):
    first = list(set(main))
    first.sort()
    out = pd.DataFrame(index=range(len(list(first))), columns=["Main"], data=list(first))
    diff_countries = []

    for column in in_data.columns:
        second = set(in_data.loc[:, column])
        both = set(first).intersection(second)
        # Necessary for some reason because a nan value managed to slip through in the Fragility dataset
        unique = [x for x in list(second - both) if type(x) == str]
        unique.sort()
        diff_countries.extend(unique)

    diff_countries = list(set(diff_countries))
    diff_countries.sort()
    i = 0
    for ctry in diff_countries:
        out.loc[i, "Non-matching"] = ctry
        i += 1
    if write:
        out.loc[:, ["Non-matching", "Main"]].to_csv("country_names.csv", index=False)


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
def format_elecsys(in_elec, in_index):
    out_data = pd.DataFrame(index=in_index, columns=["Elec_sys"])
    for _, row in in_elec.iterrows():
        out_data.loc[(row["Country"], row["Year"]), "Elec_sys"] = row["Electoral system family"]
    out_data = out_data.groupby("Country").transform(lambda group: group.ffill())
    # Unlike in most datasets, missing data here in all likelihood signifies the absence of elections, something
    # that should very much be included in the analysis
    out_data.fillna(value="No data", inplace=True)
    return out_data


def calc_rel_frag(in_reldata, in_index, first_group="Other"):
    out_data = pd.DataFrame(index=in_index, columns=["Religious fragmentation"])
    for _, row in in_reldata.iterrows():
        frag = 1
        for value in row.loc[first_group:]:
            frag -= (value/100)**2
        ctry = row.loc["Country"]
        year = row.loc["Year"]
        out_data.loc[(ctry, year), "Religious fragmentation"] = frag
    return out_data


def country_dict():
    source = pd.read_csv("concordance_table.csv")
    out = {}
    for _, row in source.iterrows():
        out[row.loc["Non-matching"]] = row.loc["Rename"]
    return out


# Transforms a dataset with country and year in different columns (with one datapoint per row)
# into a DataFrame with the standardised Multi-Index
# var_name is the name to be assigned to the new column in the output set,
# column_name the column to be searched for in the input set
def generic_list_transform(in_data, in_index, var_name, column_name=None, year_name="Year", ctry_name="Country"):
    print(var_name, ':')
    totalrows = len(in_data.index)
    done = 0
    print_freq = int(totalrows / 10)

    if column_name is None:
        column_name = var_name
    out = pd.DataFrame(index=in_index, columns=[var_name])
    for _, row in in_data.iterrows():
        if not done % print_freq:
            print(100 * done / totalrows, '%')
        ctry = row.loc["Country"]
        year = row.loc["Year"]
        slice1 = in_data.loc[in_data[year_name] == year, :]
        slice2 = slice1.loc[in_data[ctry_name] == ctry, :]
        # assert(len(slice2.loc[:, column_name].values <= 1) and f"Error in generic_list_transform(): more than one value detected for cell {ctry}, {year} in {var_name}")
        value = slice2.loc[:, column_name].values[0]
        out.loc[(ctry, year), var_name] = value
        done += 1
    return out


# Transforms a dataset with country as the index and one column per year
# into a DataFrame with the standardised Multi-Index
# var_name is the name to be assigned to the new column in the output set, no effect on input set search
def generic_table_transform(in_data, in_index, var_name, ctry_name="Country"):
    print(var_name, ':')
    totalrows = len(in_data.index)
    done = 0
    print_freq = int(totalrows / 10)

    out = pd.DataFrame(index=in_index, columns=[var_name])
    years = [col for col in in_data.columns if col.isdigit()]

    for _, row in in_data.iterrows():
        if not done % print_freq:
            print(100 * done / totalrows, '%')

        country = row.loc[ctry_name]
        for year in in_index.get_level_values(1):
            if str(year) in years:
                out.loc[(country, year), var_name] = row.loc[str(year)]
        done += 1
    return out


# Uses a country name dict to homogenise the country names. Carried out in place on the dataset given as input.
# in_index should be set to true if the country names are in the index and not in a dedicated column.
# drop_missing regulates whether countries that do not appear in the dict
# (not present in GTD, too small, sub- and super-national entities)
# should be dropped (True) or left in with their original names (False)
def rename_countries(io_data, in_dict, ctry_name="Country", in_index=False, drop_missing=True):
    if in_index:
        for country, row in io_data.iterrows():
            if country in in_dict.keys():
                newname = in_dict[country]
                if newname == "None" and drop_missing:
                    io_data.drop(country)
                elif newname == "Region" and drop_missing:
                    io_data.drop(country)
                else:
                    io_data.rename(index={country: newname})
    else:
        for index, row in io_data.iterrows():
            country = row.loc[ctry_name]
            if country in in_dict.keys():
                newname = in_dict[country]
                if newname == "None" and drop_missing:
                    row.loc[ctry_name] = None
                elif newname == "Region" and drop_missing:
                    row.loc[ctry_name] = None
                else:
                    row.loc[ctry_name] = newname
        if drop_missing:
            io_data.dropna(subset=ctry_name)


def dataprep(step="merge"):
    path_rawdata = "datasets_input/"

    path_GTD_raw = path_rawdata + "GTD_raw.csv"
    path_GTD = path_rawdata + "GTD_formatted.csv"
    path_fragility = path_rawdata + "fragility.csv"
    path_durability = path_rawdata + "dur_dem.csv"
    path_elecsys = path_rawdata + "electoral_system.csv"
    path_democracy = path_rawdata + "dur_dem.csv"
    path_FH = path_rawdata + "FH_data.csv"
    path_inequality = path_rawdata + "inequality.csv"
    path_poverty = path_rawdata + "poverty.csv"
    path_inflation = path_rawdata + "inflation.csv"
    path_lit = path_rawdata + "literacy_rate.csv"
    path_iusers = path_rawdata + "internet_users.csv"
    path_interventions = path_rawdata + "interventions.csv"
    path_religion = path_rawdata + "religion.csv"
    path_glob = path_rawdata + "globalisation.csv"

    # raw_GTD = cut_GTD(path_GTD_raw, path_GTD)
    raw_GTD = pd.read_csv(path_GTD, encoding="cp1252").rename(str.capitalize, axis="columns")
    raw_fragility = pd.read_csv(path_fragility).loc[:, ["country", "year", "sfi"]].rename(columns={"sfi" : "Fragility"}).rename(str.capitalize, axis="columns")
    raw_durability = pd.read_csv(path_durability).loc[:, ["country", "year", "durable"]].rename(str.capitalize, axis="columns")
    raw_durability = raw_durability.loc[raw_durability["Year"] >= 1950, :]
    raw_elecsys = pd.read_csv(path_elecsys).loc[:, ["Country", "Year", "Electoral system family"]].rename(str.capitalize, axis="columns")
    raw_democracy = pd.read_csv(path_democracy).loc[:, ["country", "year", "polity2"]].rename(columns={"polity2": "Democracy"}).rename(str.capitalize, axis="columns")
    raw_democracy = raw_democracy.loc[raw_democracy["Year"] >= 1950, :]
    raw_FH = pd.read_csv(path_FH, header=[0, 1, 2], index_col=0, encoding="cp1252")
    raw_inequality = pd.read_csv(path_inequality).rename(columns={"Country Name": "Country"})
    raw_poverty = pd.read_csv(path_poverty).rename(columns={"Country Name": "Country"})
    raw_poverty = raw_poverty.loc[raw_poverty["Indicator Name"] == "Poverty gap at $6.85 a day (2017 PPP) (%)", :]
    raw_inflation = pd.read_csv(path_inflation, encoding="cp1252")
    dict_lit = {"Entity": "Country", "Literacy rate, adult total (% of people ages 15 and above)": "Literacy"}
    raw_lit = pd.read_csv(path_lit).rename(columns=dict_lit).loc[:, ["Country", "Year", "Literacy"]].rename(str.capitalize, axis="columns")
    raw_iusers = pd.read_csv(path_iusers).rename(columns={"Country Name": "Country"}).rename(str.capitalize, axis="columns")
    raw_interventions = pd.read_csv(path_interventions, encoding="cp1252").loc[:, ["YEAR", "GOVTPERM", "INTERVEN1"]].rename(columns={"INTERVEN1": "Country"}).rename(str.capitalize, axis="columns")
    raw_religion = pd.read_csv(path_religion).rename(str.capitalize, axis="columns")
    raw_glob = pd.read_csv(path_glob, encoding="cp1252").loc[:, ["country", "year", "KOFGI"]].rename(columns={"KOFGI": "Globalization"}).rename(str.capitalize, axis="columns")

    main_index_ctry = raw_GTD.loc[:, "Country"].unique()
    main_index_ctry.sort()
    main_index_year = range(raw_GTD.loc[:, "Year"].min(), raw_GTD.loc[:, "Year"].max() + 1)
    main_index = pd.MultiIndex.from_product([main_index_ctry, main_index_year], names=["Country", "Year"])
    # main_data = pd.DataFrame(index=main_index)
    # main_data = format_GTD(raw_GTD, main_index)
    # main_data.to_csv(path_rawdata + "GTD_formatted.csv")
    main_data = raw_GTD

    if step == "create_dict":
        cntry_names = pd.DataFrame()
        cntry_names["Fragility"] = raw_fragility.loc[:, "Country"].unique()
        list_countries_per_set(raw_durability,"Durability", cntry_names)
        list_countries_per_set(raw_elecsys, "Election system", cntry_names)
        list_countries_per_set(raw_democracy, "Democracy", cntry_names)
        list_countries_per_set(raw_FH, "FreedomHouse", cntry_names, in_index=True)
        list_countries_per_set(raw_inequality, "Inequality", cntry_names)
        list_countries_per_set(raw_poverty, "Poverty", cntry_names)
        list_countries_per_set(raw_inflation, "Inflation", cntry_names)
        list_countries_per_set(raw_lit, "Literacy", cntry_names)
        list_countries_per_set(raw_iusers, "Internet users", cntry_names)
        list_countries_per_set(raw_interventions, "Interventions", cntry_names)
        list_countries_per_set(raw_religion, "Religious fragmentation", cntry_names)
        list_countries_per_set(raw_glob, "Globalisation", cntry_names)
        create_country_table(main_index.get_level_values(0), cntry_names, write=False)

    elif step == "merge":
        concordance_table = pd.read_csv("concordance_table.csv").loc[:, ["Non-matching", "Rename"]]
        rename_countries(raw_fragility, concordance_table)
        rename_countries(raw_durability, concordance_table)
        rename_countries(raw_elecsys, concordance_table)
        rename_countries(raw_democracy, concordance_table)
        rename_countries(raw_FH, concordance_table, in_index=True)
        rename_countries(raw_inequality, concordance_table)
        rename_countries(raw_poverty, concordance_table)
        rename_countries(raw_inflation, concordance_table)
        rename_countries(raw_lit, concordance_table)
        rename_countries(raw_iusers, concordance_table)
        rename_countries(raw_interventions, concordance_table)
        rename_countries(raw_religion, concordance_table)
        rename_countries(raw_glob, concordance_table)

        slice_fragility = generic_list_transform(raw_fragility, main_index, "Fragility")
        slice_durability = generic_list_transform(raw_durability, main_index, "Durability", column_name="Durable")
        slice_elecsys = format_elecsys(raw_elecsys, main_index)
        slice_democracy = generic_list_transform(raw_democracy, main_index, "Democracy")
        # FH data TBA
        slice_inequality = generic_table_transform(raw_inequality, main_index, "Inequality")
        slice_poverty = generic_table_transform(raw_poverty, main_index, "Poverty")
        slice_inflation = generic_table_transform(raw_inflation, main_index, "Inflation")
        slice_lit = generic_list_transform(raw_lit, main_index, "Literacy")
        slice_iusers = generic_table_transform(raw_iusers, main_index, "Internet users")
        # Interventions TBA
        slice_rel_frag = calc_rel_frag(raw_religion, main_index)
        slice_glob = generic_list_transform(raw_glob, main_index, "Globalization")

        main_data = main_data.merge(slice_fragility, left_index=True, right_index=True)
        main_data = main_data.merge(slice_durability, left_index=True, right_index=True)
        main_data = main_data.merge(slice_elecsys, left_index=True, right_index=True)
        main_data = main_data.merge(slice_democracy, left_index=True, right_index=True)
        # FH data TBA
        main_data = main_data.merge(slice_inequality, left_index=True, right_index=True)
        main_data = main_data.merge(slice_poverty, left_index=True, right_index=True)
        main_data = main_data.merge(slice_inflation, left_index=True, right_index=True)
        main_data = main_data.merge(slice_lit, left_index=True, right_index=True)
        main_data = main_data.merge(slice_iusers, left_index=True, right_index=True)
        # Interventions TBA
        main_data = main_data.merge(slice_rel_frag, left_index=True, right_index=True)
        main_data = main_data.merge(slice_glob, left_index=True, right_index=True)

        main_data.to_csv("merged_data.csv")