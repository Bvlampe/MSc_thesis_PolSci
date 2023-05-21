import pandas as pd
import numpy as np


# Cuts unnecessary data from the 100MB GTD and exports the 10x smaller dataframe back to a .csv file
def cut_GTD(path_in, path_out, code="cp1252"):
    in_data = pd.read_csv(path_in, encoding=code)
    in_data = in_data[["eventid", "iyear", "country", "country_txt", "success", "INT_LOG", "INT_IDEO"]]
    in_data.to_csv(path_out)
    return in_data


# Same but for the COW trade dataset
def cut_trade(path_in, path_out):
    in_data = pd.read_csv(path_in)
    out_data = in_data.query("ccode1 == 2 or ccode2 == 2")
    out_data.to_csv(path_out, index=False)
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
        found_any = False
        frag = 1
        for value in row.loc[first_group:]:
            if not np.isnan(value):
                found_any = True
                frag -= (value/100)**2
        ctry = row.loc["Country"]
        year = row.loc["Year"]
        if found_any:
            out_data.loc[(ctry, year), "Religious fragmentation"] = frag
    return out_data

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


def format_interventions(in_data, in_index, in_groups):
    in_data.dropna(subset="Country", inplace=True)
    out_data = pd.DataFrame(index=in_index, columns=["Intervention", "Group Intervention"])
    for _, row in in_data.iterrows():
        # Handle group interventions, continue statement avoids an attempt of individual attribution
        # of the intervention to a non-existing country with the group name
        if row.loc["Country"] in in_groups.index.get_level_values(0):
            if row.loc["Year"] in in_groups.index.get_level_values(1) and \
                    row.loc["Year"] in in_index.get_level_values(1):
                for country in in_groups.columns:
                    if in_groups.loc[(row.loc["Country"], int(row.loc["Year"])), country] == 1:
                        out_data.loc[(country, row.loc["Year"]), "Group Intervention"] = True
            continue
        if row.loc["Year"] in in_index.get_level_values(1):
            out_data.loc[(row.loc["Country"], row.loc["Year"]), "Intervention"] = True
    out_data.fillna(value=False, inplace=True)
    out_data.to_csv("test.csv")
    return out_data

def format_FH(in_data, in_index):
    out_data = pd.DataFrame(index=in_index, columns=["FH_pol", "FH_civ"])
    last_year = ""
    for column in in_data:
        var = column[1].strip()

        if var == "Status":
            continue
        colvals = in_data.loc[:, column]
        if "Unnamed" not in column[0]:
            last_year = column[0]
            # The columns with indicated years starting with "August" or "November"
            # are attributed to the following year
            if last_year[0] in ["A", "N"]:
                last_year = last_year[-4:]

        if last_year.isnumeric() and int(last_year) in in_index.get_level_values(1):
            for country, row in in_data.iterrows():
                if var == "PR":
                    out_data.loc[(country, int(last_year)), "FH_pol"] = row[column]
                elif var == "CL":
                    out_data.loc[(country, int(last_year)), "FH_civ"] = row[column]
        # This is the only column attributed to two years, as it encompasses the entirety of 1981
        # and the majority of 1982
        elif last_year == "Jan.1981-Aug. 1982":
            if var == "PR":
                out_data.loc[(country, 1981), "FH_pol"] = row[column] if '-' not in row[column] else np.nan
                out_data.loc[(country, 1982), "FH_pol"] = row[column] if '-' not in row[column] else np.nan
            elif var == "CL":
                out_data.loc[(country, 1981), "FH_civ"] = row[column] if '-' not in row[column] else np.nan
                out_data.loc[(country, 1982), "FH_civ"] = row[column] if '-' not in row[column] else np.nan
    out_data.replace(to_replace={"FH_civ": '-', "FH_pol": '-'}, value=np.nan, inplace=True)
    return out_data

def format_trade(in_data, in_index, econ_data):
    out_data = pd.DataFrame(index=in_index, columns=["US Trade"])
    for _, row in in_data.iterrows():
        ctry = row.loc["Country"]
        year = row.loc["Year"]
        if ctry in in_index.get_level_values(0) and year in in_index.get_level_values(1):
            abs_trade = row.loc["Flow1"] + row.loc["Flow2"]
            rel_trade = abs_trade * 1000 / econ_data.loc[(ctry, year), "GDP"]
            out_data.loc[(ctry, year), "US Trade"] = rel_trade
    return out_data


# Only dataprep function that already imputes data (all other interpolation is done leading up to the model creation,
# but is not written to the merged dataset file
# This is because it is necessary for this dataset to interpolate missing variables before merging it, as multiple
# entries per country may exist and if some, but not all of these entries are missing data for a specific year, summing
# them up will result in non-missing data, which will circumvent interpolation and potentially lead to grossly
# misleading numbers
def format_weapons(in_data, in_index):
    out_data = pd.DataFrame(index=in_index, columns=["Weapon imports"])
    for ctry, row in in_data.iterrows():
        ip_row = row.interpolate(limit=10, limit_area="inside")
        for year in row.index:
            if (ctry not in in_index.get_level_values(0)) or (int(year) not in in_index.get_level_values(1)):
                continue
            if np.isnan(out_data.loc[(ctry, int(year)), "Weapon imports"]):
                out_data.loc[(ctry, int(year)), "Weapon imports"] = ip_row[year]
            else:
                out_data.loc[(ctry, int(year)), "Weapon imports"] += ip_row[year]
    return out_data
