from collections import defaultdict

import numpy as pd
import pandas as pd



STATNAMES = ["Score","FGM","FGA","FGM3","FGA3","FTM","FTA","OR","DR","Ast","TO","Stl","Blk","PF"]

def partition_games(df, frac=0.7):
    """Returns a 2-tuple of arrays containing the indices.
       The first are the indices in the dataframe that correspond to the
       first <frac> games of each season. The second are the indices
       that correspond to the last <1-frac> games of each season.
       Assumes that the games are already sorted into chronological order.
    """
    years = sorted(df.Season.unique())
    first, second = [], []
    for year in years:
        idcs = list(df.index[df.Season==year])
        thresh = int(len(idcs)*frac)
        first += idcs[:thresh]
        second += idcs[thresh:]
        
    return first, second


def compute_season_stats(df):
    """Take the per-game data frame and aggregate stats on a per team/season basis
       Returns a dict structured like stats[year][team_id][stat_name]
    """
    if "poss" not in df.columns:
        df["poss"] = 0.5*(df.WFGA + 0.44*df.WFTA - df.WOR + df.WTO + df.LFGA + 0.44*df.LFTA - df.LOR + df.LTO)
    
    stats = {}
    for irow,row in df.iterrows():
        year = row.Season
        wid = row.WTeamID
        lid = row.LTeamID
        if year not in stats:
            stats[year] = {}
       
        # initialize values if we haven't seen this team yet
        for tid in (wid,lid):
            if tid not in stats[year]:
                stats[year][tid] = {}
                stats[year][tid]["wins"] = 0
                stats[year][tid]["losses"] = 0
                stats[year][tid]["totOT"] = 0
                stats[year][tid]["opps"] = []        
                stats[year][tid]["scores"] = []                
                stats[year][tid]["poss"] = []                
                stats[year][tid]["nOT"] = []                
                for sn in STATNAMES:
                    stats[year][tid]["T"+sn] = 0
                    stats[year][tid]["O"+sn] = 0
        
        for sn in STATNAMES:
            stats[year][wid]["T"+sn] += row["W"+sn]
            stats[year][wid]["O"+sn] += row["L"+sn]
            stats[year][lid]["T"+sn] += row["L"+sn]
            stats[year][lid]["O"+sn] += row["W"+sn]
        stats[year][wid]["wins"] += 1
        stats[year][lid]["losses"] += 1
        stats[year][wid]["totOT"] += row.NumOT
        stats[year][lid]["totOT"] += row.NumOT
        stats[year][wid]["opps"].append(lid)
        stats[year][lid]["opps"].append(wid)
        stats[year][wid]["scores"].append((row.WScore,row.LScore))
        stats[year][lid]["scores"].append((row.LScore,row.WScore))
        stats[year][wid]["poss"].append(row.poss)
        stats[year][lid]["poss"].append(row.poss)
        stats[year][wid]["nOT"].append(row.NumOT)
        stats[year][lid]["nOT"].append(row.NumOT)

    return stats

def stats_dict_to_df(stats):
    """Convert a dict of aggregated season stats (as returned by compute_season_stats)
       into a DataFrame, with one team/season pair per row
    """    
    ascols = defaultdict(list)
    for year in sorted(stats.keys()):
        for tid in sorted(stats[year].keys()):
            ascols["year"].append(year)
            ascols["team_id"].append(tid)
            for s in stats[year][tid].keys():
                ascols[s].append(stats[year][tid][s])

    return pd.DataFrame(ascols, 
                        columns=["year","team_id","wins","losses","totOT"] + \
                        ["T"+sn for sn in STATNAMES] + \
                        ["O"+sn for sn in STATNAMES])

def stats_df_to_dict(df):
    """Convert a DataFrame of aggregated season stats
       into a dict structured as stats[year][team_id][stat_name]
    """ 
    stats = {}
    for irow, row in df.iterrows():
        if row.year not in stats:
            stats[row.year] = {}
        if row.team_id not in stats[row.year]:
            stats[row.year][row.team_id] = {}
        for col in df.columns[2:]:
            stats[row.year][row.team_id][col] = row[col]
    return stats

def add_advanced_stats(df):
    """ Add some advanced stats to a season stats dataframe """
    # compute advanced stats
    for c in ('T','O'):
        df[c+'poss'] = df[c+"FGA"] + 0.44*df[c+"FTA"] - df[c+"OR"] + df[c+"TO"]
        df[c+'eff'] = 100. * df[c+"Score"] / df[c+"poss"]
        df[c+'astr'] = df[c+"Ast"] / (df[c+"FGA"] + 0.44*df[c+"FTA"] + df[c+"Ast"] + df[c+"TO"])
        df[c+'tovr'] = df[c+"TO"] / (df[c+"FGA"] + 0.44*df[c+"FTA"] + df[c+"TO"])
        df[c+'efgp'] = (df[c+"FGM"] + 0.5*df[c+"FGM3"]) / df[c+"FGA"]
        df[c+'orbp'] = df[c+'OR'] / (df[c+'OR'] + df[c+'DR'])
        df[c+'ftr'] = df[c+"FTA"] / df[c+"FGA"]
    df['rawpace'] = 0.5*(df["Tposs"]+df["Oposs"]) / (df["wins"] + df["losses"] + 0.125*df["totOT"])

