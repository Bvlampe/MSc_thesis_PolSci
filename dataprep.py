import pandas as pd
import numpy as np
import var_edits
import time


def query_yn(start_time=None, query="Write to file? y/n: "):
    if start_time is not None:
        print("Elapsed time at query_yn():", time.time() - start_time, "sec")
    answer = ""
    while answer not in ["y", "n"]:
        answer = input(query)
    return True if answer == "y" else False


def list_countries_per_set(in_dataset, name_dataset, io_list, name_column="Country", in_index=False, in_header=False):
    assert((not(in_header and in_index)) and "Error: list_countries_per_set() was called with both in_index and in_header set as True")
    i = 0
    countries = []
    if in_index:
        countries = in_dataset.index.unique()
    elif in_header:
        countries = in_dataset.columns.unique()
    else:
        countries = in_dataset.loc[:, name_column].unique()
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

def update_table(main, in_data):
    old_table = pd.read_csv("concordance_table.csv")
    first = list(set(main))
    first.sort()
    out = pd.DataFrame(index=range(len(list(first))), columns=["Main", "Non-matching"])
    out.loc[:, "Main"] = list(first)
    diff_countries = []

    for column in in_data.columns:
        second = set(in_data.loc[:, column])
        both = set(first).intersection(second)
        # Necessary for some reason because a nan value managed to slip through in the Fragility dataset
        unique = [x for x in list(second - both) if type(x) == str]
        unique.sort()
        for newctry in unique:
            if newctry not in old_table.loc[:, "Non-matching"].values:
                diff_countries.append(newctry)

    diff_countries = list(set(diff_countries))
    diff_countries.sort()
    i = 0
    for ctry in diff_countries:
        out.loc[i, "Non-matching"] = ctry
        i += 1
    print(out)
    print("Number of new non-matching:", len(out.loc[:,"Non-matching"].unique()) - 1)
    if query_yn():
        out.loc[:, ["Non-matching", "Main"]].to_csv("updated_country_names.csv", index=False)


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


def rename_countries(io_data, in_dict, ctry_name="Country", in_index=False, in_header=False,
                     drop_missing=True, leave_groups=False):
    assert ((not (
                in_header and in_index)) and "Error: rename_countries() was called with both in_index and in_header set as True")
    if in_index:
        for country in io_data.index:
            if country in in_dict.keys():
                newname = in_dict[country]
                if newname == "None" and drop_missing:
                    io_data.drop(country, inplace=True)
                elif newname == "Region" and drop_missing:
                    io_data.drop(country, inplace=True)
                else:
                    io_data.rename(index={country: newname}, inplace=True)
    elif in_header:
        for country in io_data.columns:
            if country in in_dict.keys():
                newname = in_dict[country]
                if newname == "None" and drop_missing:
                    io_data.drop(country, inplace=True, axis=1)
                elif newname == "Region" and drop_missing:
                    io_data.drop(country, inplace=True, axis=1)
                else:
                    io_data.rename(columns={country: newname}, inplace=True)

    else:
        for i in range(len(io_data.index)):
            country = io_data.loc[i, ctry_name]
            if leave_groups and (country in ["Arab League", "AU", "CIS", "ECOWAS", "EU", "NATO", "UN"]):
                continue
            if country in in_dict.keys():
                newname = in_dict[country]
                if newname == "None" and drop_missing:
                    io_data.loc[i, ctry_name] = None
                elif newname == "Region" and drop_missing:
                    io_data.loc[i, ctry_name] = None
                else:
                    io_data.loc[i, ctry_name] = newname
        if drop_missing:
            io_data.dropna(subset=ctry_name, inplace=True)


def dataprep(step="merge", edit_col=None):
    start_time = time.time()

    path_rawdata = "datasets_input/"

    # path_GTD_raw = path_rawdata + "GTD_raw.csv"
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
    path_edu = path_rawdata + "education.csv"
    path_econ = path_rawdata + "econ.csv"
    path_pop = path_rawdata + "population.csv"
    path_groups = path_rawdata + "group_membership.csv"
    # path_trade_raw = path_rawdata + "Dyadic_COW_4.0.csv"
    path_trade = path_rawdata + "trade.csv"

    # Only done once
    # raw_GTD = cut_GTD(path_GTD_raw, path_GTD)
    # raw_trade = var_edits.cut_trade(path_trade_raw, path_trade)
    raw_GTD = pd.read_csv(path_GTD, encoding="cp1252").rename(str.capitalize, axis="columns")
    raw_fragility = pd.read_csv(path_fragility).loc[:, ["country", "year", "sfi"]].rename(columns={"sfi" : "Fragility"}).rename(str.capitalize, axis="columns")
    raw_durability = pd.read_csv(path_durability).loc[:, ["country", "year", "durable"]].rename(str.capitalize, axis="columns")
    raw_durability = raw_durability.loc[raw_durability["Year"] >= 1950, :].reset_index()
    raw_elecsys = pd.read_csv(path_elecsys).loc[:, ["Country", "Year", "Electoral system family"]].rename(str.capitalize, axis="columns")
    raw_democracy = pd.read_csv(path_democracy).loc[:, ["country", "year", "polity2"]].rename(columns={"polity2": "Democracy"}).rename(str.capitalize, axis="columns")
    raw_democracy = raw_democracy.loc[raw_democracy["Year"] >= 1950, :].reset_index()
    raw_FH = pd.read_csv(path_FH, header=[0, 1], index_col=0, encoding="cp1252")
    raw_inequality = pd.read_csv(path_inequality).rename(columns={"Country Name": "Country"})
    raw_poverty = pd.read_csv(path_poverty).rename(columns={"Country Name": "Country"})
    raw_poverty = raw_poverty.loc[raw_poverty["Indicator Name"] == "Poverty gap at $6.85 a day (2017 PPP) (%)", :].reset_index()
    raw_inflation = pd.read_csv(path_inflation, encoding="cp1252")
    dict_lit = {"Entity": "Country", "Literacy rate, adult total (% of people ages 15 and above)": "Literacy"}
    raw_lit = pd.read_csv(path_lit).rename(columns=dict_lit).loc[:, ["Country", "Year", "Literacy"]].rename(str.capitalize, axis="columns")
    raw_iusers = pd.read_csv(path_iusers).rename(columns={"Country Name": "Country"}).rename(str.capitalize, axis="columns")
    raw_interventions = pd.read_csv(path_interventions, encoding="cp1252").loc[:, ["YEAR", "GOVTPERM", "INTERVEN1"]].rename(columns={"INTERVEN1": "Country"}).rename(str.capitalize, axis="columns")
    raw_religion = pd.read_csv(path_religion).rename(str.capitalize, axis="columns")
    raw_glob = pd.read_csv(path_glob, encoding="cp1252").loc[:, ["country", "year", "KOFGI"]].rename(columns={"KOFGI": "Globalization"}).rename(str.capitalize, axis="columns")
    raw_edu = pd.read_csv(path_edu).loc[:, ["country", "year", "lpc"]].rename(columns={"lpc": "education"}).rename(str.capitalize, axis="columns")
    raw_econ = pd.read_csv(path_econ, encoding="cp1252").loc[:, ["country", "year", "rgdpe"]].rename(str.capitalize, axis="columns").rename(columns={"Rgdpe": "GDP"})
    raw_pop = pd.read_csv(path_pop).rename(columns={"Country Name": "Country"})
    raw_groups = pd.read_csv(path_groups, index_col=[0, 1]).rename(columns={"ioname": "Group"}).rename(str.capitalize, axis="columns")
    raw_trade = pd.read_csv(path_trade).rename(str.capitalize, axis="columns").rename(columns={"Importer2": "Country"}).replace({-9: np.nan})

    main_index_ctry = raw_GTD.loc[:, "Country"].unique()
    main_index_ctry.sort()
    main_index_year = range(raw_GTD.loc[:, "Year"].min(), raw_GTD.loc[:, "Year"].max() + 1)
    main_index = pd.MultiIndex.from_product([main_index_ctry, main_index_year], names=["Country", "Year"])

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
        list_countries_per_set(raw_edu, "Education", cntry_names)
        list_countries_per_set(raw_econ, "GDP", cntry_names)
        list_countries_per_set(raw_pop, "Population", cntry_names)
        list_countries_per_set(raw_groups, "Groups", cntry_names, in_header=True)
        list_countries_per_set(raw_trade, "Trade", cntry_names)
        create_country_table(main_index.get_level_values(0), cntry_names, write=query_yn(start_time))

    elif step == "update_dict":
        cntry_names = pd.DataFrame()
        cntry_names["Trade"] = raw_trade.loc[:, "Country"].unique()
        update_table(main_index.get_level_values(0), cntry_names)

    elif step == "merge":
        # main_data = pd.DataFrame(index=main_index)
        # main_data = format_GTD(raw_GTD, main_index)
        # main_data.to_csv(path_rawdata + "GTD_formatted.csv")

        concordance_table = country_dict()
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
        rename_countries(raw_interventions, concordance_table, leave_groups=True)
        rename_countries(raw_religion, concordance_table)
        rename_countries(raw_glob, concordance_table)
        rename_countries(raw_edu, concordance_table)
        rename_countries(raw_econ, concordance_table)
        rename_countries(raw_pop, concordance_table)
        rename_countries(raw_groups, concordance_table, in_header=True)
        rename_countries(raw_trade, concordance_table)

        main_data = generic_list_transform(raw_GTD, main_index, "Terrorist attack")
        slice_fragility = generic_list_transform(raw_fragility, main_index, "Fragility")
        slice_durability = generic_list_transform(raw_durability, main_index, "Durability", column_name="Durable")
        slice_elecsys = var_edits.format_elecsys(raw_elecsys, main_index)
        slice_democracy = generic_list_transform(raw_democracy, main_index, "Democracy")
        slice_FH = var_edits.format_FH(raw_FH, main_index)
        slice_inequality = generic_table_transform(raw_inequality, main_index, "Inequality")
        slice_poverty = generic_table_transform(raw_poverty, main_index, "Poverty")
        slice_inflation = generic_table_transform(raw_inflation, main_index, "Inflation")
        slice_lit = generic_list_transform(raw_lit, main_index, "Literacy")
        slice_iusers = generic_table_transform(raw_iusers, main_index, "Internet users")
        slice_interventions = var_edits.format_interventions(
            raw_interventions, main_index, raw_groups.rename({"LOAS": "Arab League"}, level=0))
        slice_rel_frag = var_edits.calc_rel_frag(raw_religion, main_index)
        slice_glob = generic_list_transform(raw_glob, main_index, "Globalization")
        slice_edu = generic_list_transform(raw_edu, main_index, "Education")
        slice_econ = generic_list_transform(raw_econ, main_index, "GDP")
        slice_pop = generic_table_transform(raw_pop, main_index, "Population")
        slice_trade = var_edits.format_trade(raw_trade, main_index, slice_econ)

        main_data = main_data.merge(slice_fragility, left_index=True, right_index=True)
        main_data = main_data.merge(slice_durability, left_index=True, right_index=True)
        main_data = main_data.merge(slice_elecsys, left_index=True, right_index=True)
        main_data = main_data.merge(slice_democracy, left_index=True, right_index=True)
        main_data = main_data.merge(slice_FH, left_index=True, right_index=True)
        main_data = main_data.merge(slice_inequality, left_index=True, right_index=True)
        main_data = main_data.merge(slice_poverty, left_index=True, right_index=True)
        main_data = main_data.merge(slice_inflation, left_index=True, right_index=True)
        main_data = main_data.merge(slice_lit, left_index=True, right_index=True)
        main_data = main_data.merge(slice_iusers, left_index=True, right_index=True)
        main_data = main_data.merge(slice_interventions, left_index=True, right_index=True)
        main_data = main_data.merge(slice_rel_frag, left_index=True, right_index=True)
        main_data = main_data.merge(slice_glob, left_index=True, right_index=True)
        main_data = main_data.merge(slice_edu, left_index=True, right_index=True)
        main_data = main_data.merge(slice_econ, left_index=True, right_index=True)
        main_data = main_data.merge(slice_pop, left_index=True, right_index=True)
        main_data = main_data.merge(slice_trade, left_index=True, right_index=True)
        main_data.loc[:, "GDP_pp"] = (main_data.loc[:, "GDP"] * 1000000) / main_data.loc[:, "Population"]

        if query_yn(start_time):
            main_data.to_csv("merged_data.csv")

    # Not needed in common usage, only for short ad-hoc patches
    # (all patches should also be integrated into "merge" mode for potential future full re-runs)
    # Only rel_frag and intervention editing implemented, to be expanded as needed
    elif step == "edit":
        assert(edit_col is not None)
        main_data = pd.read_csv("merged_data.csv", index_col=[0, 1])

        if edit_col == "Religious fragmentation":
            slice_rel_frag = var_edits.calc_rel_frag(raw_religion, main_index)
            main_data.loc[:, "Religious fragmentation"] = slice_rel_frag.loc[:, "Religious fragmentation"]

        elif edit_col == "set_1":
            concordance_table = country_dict()
            rename_countries(raw_edu, concordance_table)
            rename_countries(raw_econ, concordance_table)
            rename_countries(raw_pop, concordance_table)
            slice_edu = generic_list_transform(raw_edu, main_index, "Education")
            slice_econ = generic_list_transform(raw_econ, main_index, "GDP")
            slice_pop = generic_table_transform(raw_pop, main_index, "Population")
            main_data = main_data.merge(slice_edu, left_index=True, right_index=True)
            main_data = main_data.merge(slice_econ, left_index=True, right_index=True)
            main_data = main_data.merge(slice_pop, left_index=True, right_index=True)

        elif edit_col == "FH":
            concordance_table = country_dict()
            rename_countries(raw_FH, concordance_table, in_index=True)
            slice_FH = var_edits.format_FH(raw_FH, main_index)
            main_data.loc[:, "FH_civ"] = slice_FH.loc[:, "FH_civ"]
            main_data.loc[:, "FH_pol"] = slice_FH.loc[:, "FH_pol"]
            # main_data = main_data.merge(slice_FH, left_index=True, right_index=True)
            main_data.replace(to_replace={"FH_civ": '-', "FH_pol": '-'}, value=np.nan, inplace=True)

        elif edit_col == "GDP_pp":
            main_data.loc[:, "GDP_pp"] = (main_data.loc[:, "GDP"] * 1000000) / main_data.loc[:, "Population"]

        elif edit_col == "Interventions":
            concordance_table = country_dict()
            rename_countries(raw_interventions, concordance_table, leave_groups=True)
            rename_countries(raw_groups, concordance_table, in_header=True)
            slice_interventions = var_edits.format_interventions(
                raw_interventions, main_index, raw_groups.rename({"LOAS": "Arab League"}, level=0))
            main_data.drop("Intervention", axis=1, inplace=True)
            main_data = main_data.merge(slice_interventions, left_index=True, right_index=True)

        elif edit_col == "Trade":
            concordance_table = country_dict()
            rename_countries(raw_trade, concordance_table)
            slice_trade = var_edits.format_trade(raw_trade, main_index, main_data)
            main_data = main_data.merge(slice_trade, left_index=True, right_index=True)

        if query_yn(start_time):
            main_data.to_csv("merged_data.csv")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed:", elapsed_time, "sec")