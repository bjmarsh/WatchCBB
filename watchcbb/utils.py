import os
import pickle
import gzip
from collections import defaultdict
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd



STATNAMES = ["Score","FGM","FGA","FGM3","FGA3","FTM","FTA","OR","DR","Ast","TO","Stl","Blk","PF"]
ADVSTATNAMES = ['eff','astr','orbp','tovr','efgp','ftr']
ADVSTATFEATURES = ["T"+stat for stat in ADVSTATNAMES] + ["O"+stat for stat in ADVSTATNAMES]
PCAFEATURES = [f"PCA{i}" for i in range(len(ADVSTATFEATURES))]
RIVALRIES = [tuple(sorted([x.strip() for x in line.split(',')])) \
             for line in open(os.path.join(os.path.dirname(__file__),'../data/rivalries.txt'))]

def partition_games(df, frac=0.7):
    """
    Returns a 2-tuple of arrays containing the indices.
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

def add_gid(df):
    """ Add a column to game df with unique game id """

    def get_gid(row):
        tid1, tid2 = sorted((row.WTeamID, row.LTeamID))
        return '{0}_{1}_{2}'.format(row.Date, tid1, tid2)

    df['gid'] = df.apply(get_gid, axis=1)

def add_poss(df):
    """ Add a column to game df that represents number of possessions """
    df["poss"] = 0.5*(df["WFGA"] + 0.44*df["WFTA"] - df["WOR"] + df["WTO"] + df["LFGA"] + 0.44*df["LFTA"] - df["LOR"] + df["LTO"])

def compute_season_stats(df, df_preseason=None, force_all_teams=False, tids=None, years=None):
    """
    Take the per-game data frame and aggregate stats on a per team/season basis
    Returns a dict structured like stats[year][team_id][stat_name]
    If df_preseason is not None, also include preseason predictions of efficiency/pace
    If force_all_teams, include all teams even if they haven't played any games yet
      (if this is True, must also pass list of tids)
      If years is not None, force include all years in that list
    """

    if df_preseason is not None:
        dict_preseason = stats_df_to_dict(df_preseason)

    if force_all_teams:
        if tids is None:
            raise Exception("Must pass a df_teams if force_all_teams is True")
    
    if years is None:
        years = df.Season.unique()

    def init_team_dict(d, year, tid):
        if year not in d:
            raise Exception(f"{year} must be a key of dictionary")
        if tid in d[year]:
            raise Exception(f"team_id {tid} already initialized in dictionary")
        d[year][tid] = {}
        d[year][tid]["wins"] = 0
        d[year][tid]["losses"] = 0
        d[year][tid]["totOT"] = 0
        d[year][tid]["totPoss"] = 0
        d[year][tid]["opps"] = []
        d[year][tid]["scores"] = []
        d[year][tid]["HA"] = []
        d[year][tid]["poss"] = []
        d[year][tid]["nOT"] = []
        for sn in STATNAMES:
            d[year][tid]["T"+sn] = 0
            d[year][tid]["O"+sn] = 0
        if df_preseason is not None:
            d[year][tid]["preseason_eff"] = dict_preseason[year].get(tid,{}).get("pred_eff", -10.0)
            d[year][tid]["preseason_oeff"] = dict_preseason[year].get(tid,{}).get("pred_oeff", -10.0)
            d[year][tid]["preseason_deff"] = dict_preseason[year].get(tid,{}).get("pred_deff", -10.0)
            d[year][tid]["preseason_pace"] = dict_preseason[year].get(tid,{}).get("pred_pace", -10.0)

    stats = {}

    # initialize all years
    for year in years:
        stats[year] = {}
        # force initialization of all teams, even if they've played no games yet
        if force_all_teams:
            for tid in tids:
                init_team_dict(stats, year, tid)

    for irow,row in df.iterrows():
        year = row.Season
        wid = row.WTeamID
        lid = row.LTeamID

        # initialize values if we haven't seen this team yet
        for tid in (wid,lid):
            if tid not in stats[year]:
                init_team_dict(stats, year, tid)

        for sn in STATNAMES:
            stats[year][wid]["T"+sn] += row["W"+sn]
            stats[year][wid]["O"+sn] += row["L"+sn]
            stats[year][lid]["T"+sn] += row["L"+sn]
            stats[year][lid]["O"+sn] += row["W"+sn]
        stats[year][wid]["wins"] += 1
        stats[year][lid]["losses"] += 1
        stats[year][wid]["totOT"] += row.NumOT
        stats[year][lid]["totOT"] += row.NumOT
        stats[year][wid]["totPoss"] += row.poss
        stats[year][lid]["totPoss"] += row.poss
        stats[year][wid]["opps"].append(lid)
        stats[year][lid]["opps"].append(wid)
        stats[year][wid]["scores"].append((row.WScore,row.LScore))
        stats[year][lid]["scores"].append((row.LScore,row.WScore))
        stats[year][wid]["HA"].append(row.WLoc)
        stats[year][lid]["HA"].append('HNA'['ANH'.find(row.WLoc)])
        stats[year][wid]["poss"].append(row.poss)
        stats[year][lid]["poss"].append(row.poss)
        stats[year][wid]["nOT"].append(row.NumOT)
        stats[year][lid]["nOT"].append(row.NumOT)

    return stats


def stats_dict_to_df(stats):
    """
    Convert a dict of aggregated season stats (as returned by compute_season_stats)
    into a DataFrame, with one team/season pair per row
    """    
    ascols = defaultdict(list)
    for year in sorted(stats.keys()):
        for tid in sorted(stats[year].keys()):
            ascols["year"].append(year)
            ascols["team_id"].append(tid)
            for s in stats[year][tid].keys():
                ascols[s].append(stats[year][tid][s])

    columns = ["year","team_id","wins","losses","totOT"] + \
              ["T"+sn for sn in STATNAMES] + ["O"+sn for sn in STATNAMES]
    columns += list(set(stats[year][tid].keys())-set(columns))

    return pd.DataFrame(ascols, columns=columns)


def stats_df_to_dict(df):
    """
    Convert a DataFrame of aggregated season stats
    into a dict structured as stats[year][team_id][stat_name]
    Assumes the first two columns of df are ['year','team_id']
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
        opp = 'T' if c=='O' else 'O'
        df[c+'poss'] = df[c+"FGA"] + 0.44*df[c+"FTA"] - df[c+"OR"] + df[c+"TO"]
        df[c+'eff'] = 100. * df[c+"Score"] / df[c+"poss"]
        df[c+'astr'] = df[c+"Ast"] / (df[c+"FGA"] + 0.44*df[c+"FTA"] + df[c+"Ast"] + df[c+"TO"])
        df[c+'tovr'] = df[c+"TO"] / (df[c+"FGA"] + 0.44*df[c+"FTA"] + df[c+"TO"])
        df[c+'efgp'] = (df[c+"FGM"] + 0.5*df[c+"FGM3"]) / df[c+"FGA"]
        df[c+'orbp'] = df[c+'OR'] / (df[c+'OR'] + df[opp+'DR'])
        df[c+'ftr'] = df[c+"FTA"] / df[c+"FGA"]
    df['rawpace'] = 0.5*(df["Tposs"]+df["Oposs"]) / (df["wins"] + df["losses"] + 0.125*df["totOT"])



def compile_training_data(df, season_stats_dict, random_seed=0, sort='random', include_preseason=False):
    """
    Take in a raw game-level dataframe as well as a dictionary of season stats,
    and generate a tidy dataframe amenable to feeding into ML models.
    
    Parameters:
    - sort: if 'random', randomly choose team to be the "reference" team.
            if 'alphabetical', use the first team alphabetically
    - include_preseason: whether to add a few features on preseason statistics.
                         these must be present in season_stats_dict
    """
    np.random.seed(random_seed)
    data = defaultdict(list)
    for irow, row in df.iterrows():
        d = season_stats_dict[row.Season]
        if sort=='random':
            dowin = np.random.randint(2)
        elif sort=='alphabetical':
            dowin = (row.WTeamID < row.LTeamID)
        else:
            raise Exception("Illegal sort parameter "+sort)
        id1, id2 = row.WTeamID, row.LTeamID
        mult = (1 if dowin else -1)
        if not dowin:
            id1, id2 = id2, id1
        data['result'].append(dowin)
        data['margin'].append((row.WScore-row.LScore)*mult)
        data['totscore'].append(row.WScore+row.LScore)
        data['date'].append(row.Date)
        data['season'].append(row.Season)
        data['gid'].append(row.gid)
        data['tid1'].append(id1)
        data['tid2'].append(id2)
        data['rank1'].append(row.Wrank if dowin else row.Lrank)
        data['rank2'].append(row.Lrank if dowin else row.Wrank)
        data['poss'].append(row.poss / (1.0 + 0.125*row.NumOT))
        data['pace1'].append(d[id1]['pace'])
        data['pace2'].append(d[id2]['pace'])
        data['HA'].append(('ANH'.find(row.WLoc)-1) * mult)
        if include_preseason:
            data['preseason_effdiff'].append(d[id1]["preseason_eff"] - d[id2]["preseason_eff"])
            data['preseason_effsum'].append(d[id1]["preseason_oeff"] + d[id1]["preseason_deff"] + d[id2]["preseason_oeff"] + d[id2]["preseason_deff"])
            data['preseason_paceprod'].append(d[id1]["preseason_pace"]*d[id2]["preseason_pace"])
        data['effdiff'].append(d[id1]["Tneteff"] - d[id2]["Tneteff"])
        data['effsum'].append(d[id1]["Tcorroeff"] + d[id1]["Tcorrdeff"] + d[id2]["Tcorroeff"] + d[id2]["Tcorrdeff"])
        data['neteffsum'].append(d[id1]["Tcorroeff"] - d[id1]["Tcorrdeff"] + d[id2]["Tcorroeff"] - d[id2]["Tcorrdeff"])
        data['compratsum'].append(d[id1]["CompositeRating"] + d[id2]["CompositeRating"])
        data['raweffdiff'].append((d[id1]["Teff"] - d[id1]["Oeff"]) - \
                                   (d[id2]["Teff"] - d[id2]["Oeff"]))
        # 'T'+stat is difference in offensive stats between two teams. 'O'+stat is difference in defensive
        for stat in ADVSTATNAMES:
            data['T'+stat].append(d[id1]['Tcorro'+stat] - d[id2]['Tcorro'+stat])
            data['O'+stat].append(d[id1]['Tcorrd'+stat] - d[id2]['Tcorrd'+stat])
         
    columns = ['season', 'date', 'gid','tid1','tid2','result','rank1','rank2','totscore', 'margin', 
               'HA','poss','pace1','pace2','effdiff','raweffdiff','effsum']
    columns += list(set(data.keys()) - set(columns))
    return pd.DataFrame(data, columns=columns)


def train_test_split_by_year(data, train_years, test_years, pca_model=None):
    """ 
    From a dataframe generated by compile_training_data, split in to train and test sets by year.
    If do_pca, a PCA is fit on the 12 advance stat features,
    and the corresponding 12 components are added as features to the output dataframes.
    """
    data_train = data.loc[data.season.isin(train_years)].copy()
    data_test = data.loc[data.season.isin(test_years)].copy()
    
    if pca_model is not None:
        xf_train = pca_model.fit_transform(data_train[ADVSTATFEATURES])
        xf_test = pca_model.transform(data_test[ADVSTATFEATURES])
        for i in range(xf_train.shape[1]):
            data_train["PCA"+str(i)] = xf_train[:,i]
            data_test["PCA"+str(i)] = xf_test[:,i]
    
    return data_train, data_test

def get_daily_predictions(dates, df_allgames, model_file, pickled_stats_dir, return_cols=None, no_tqdm=False):
    """ 
    Make predictions for games on various dates, *with statistics as they were on that date*
    Used for validating performance of a model over the course of the season
    
    Parameters:
    - dates: list of dates to make predictions for
    - df_allgames: dataframe of individual games, including the dates that are requested
    - model_file: pickled sklearn models to make per-game predictions
    - pickled_stats_dir: dictionaries/dataframes of aggregated season stats should be in pickled
                         files here, one file per date
    - return cols: list of columns we want to return

    Note that three sets of predictions are made: one with only preseason predictions,
    one with only current season stats, and one using a blend of the two
    """
    
    with open(model_file, 'rb') as fid:
        pca, logreg, logreg_simple, linreg_pace, linreg_margin, linreg_total = pickle.load(fid)
    if return_cols is None:
        return_cols = ['gid','result','HA','effdiff','preseason_effdiff',
                       'prob','preseason_prob','blended_prob',
                       'poss','pred_pace','pred_pace_pre','pred_pace_blend',
                       'totscore','pred_total','pred_total_pre','pred_total_blend',
                       'margin','pred_margin','pred_margin_pre','pred_margin_blend']
    dfs = []
    total_games = defaultdict(int)
    if not no_tqdm:
        dates = tqdm_dates
    for date in dates:
        df_games = df_allgames.loc[df_allgames.Date==date]
        year = df_games.Season.values[0]
        total_games[year] += df_games.shape[0]
        try:
            with gzip.open('{0}/{1}.pkl.gz'.format(pickled_stats_dir.format(year=year), date), 'rb') as fid:
                ssd, _ = pickle.load(fid)
        except FileNotFoundError:
            continue
        games = compile_training_data(df_games, ssd, sort='alphabetical', include_preseason=True)
        games = games.loc[(abs(games.effdiff)<900) & ~games.pace1.isna()]
        if games.shape[0] == 0:
            continue
        
        xf = pca.transform(games[ADVSTATFEATURES])
        for i in range(len(ADVSTATFEATURES)):
            games["PCA"+str(i)] = xf[:,i]

        probs = logreg.predict_proba(games[PCAFEATURES + ['HA']])[:,1]
        probs_pre = logreg_simple.predict_proba(games[['preseason_effdiff','HA']])[:,1] 
        pred_pace = linreg_pace.predict(np.array(games.pace1*games.pace2).reshape(-1,1))
        pred_pace_pre = linreg_pace.predict(games[['preseason_paceprod']])
        pred_total = linreg_total.predict((pred_pace*games.effsum.values).reshape(-1,1))
        pred_total_pre = linreg_total.predict((pred_pace_pre*games.preseason_effsum.values).reshape(-1,1))
        pred_margin = linreg_margin.predict(np.array([pred_pace*games.effdiff.values, games.HA.values]).T)
        pred_margin_pre = linreg_margin.predict(np.array([pred_pace_pre*games.preseason_effdiff.values, games.HA.values]).T)

        p = get_blend_param(total_games[year] / 5400.0)
        # we want to blend the exponents, so de logify, blend, and re compute logistic
        probs_blend = 1 / (1 + np.exp(-p*(-np.log(1./probs_pre-1)) - (1-p)*(-np.log(1./probs-1))))
        # these are just linear combos, since regression is linear
        pred_pace_blend = p*pred_pace_pre + (1-p)*pred_pace
        pred_total_blend = p*pred_total_pre + (1-p)*pred_total
        pred_margin_blend = p*pred_margin_pre + (1-p)*pred_margin

        games["prob"] = probs
        games["preseason_prob"] = probs_pre
        games["blended_prob"] = probs_blend
        games['pred_pace'] = pred_pace
        games['pred_pace_pre'] = pred_pace_pre
        games['pred_pace_blend'] = pred_pace_blend
        games['pred_total'] = pred_total
        games['pred_total_pre'] = pred_total_pre
        games['pred_total_blend'] = pred_total_blend
        games['pred_margin'] = pred_margin
        games['pred_margin_pre'] = pred_margin_pre
        games['pred_margin_blend'] = pred_margin_blend

        dfs.append(games[return_cols].copy())

    if len(dfs) > 0:
        return pd.concat(dfs, ignore_index=True)
    else:
        return None


def get_pca_model():
    """PCA pipeline used for logistic regression"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA())
    ])


def is_upset(r1, r2):
    """Return true if r1 beating r2 is classified as an upset"""

    return (r1 < 0 and 0 < r2 <= 20) or (r1 > 0 and r2 > 0 and r1-r2 > 10)


def get_df_upset_prob(row):
    """Call df.apply on this to get upset probability per row. 0 if no potential upset."""
    
    if is_upset(row.rank1, row.rank2):
        return row.prob
    if is_upset(row.rank2, row.rank1):
        return 1-row.prob
    return 0.0


def is_rivalry(row):
    """True if this row contains a rivalry game as defined in data/rivalries.txt"""
    tid1, tid2 = sorted(row.gid.split('_')[1:])
    return (tid1,tid2) in RIVALRIES


def get_blend_param(season_frac):
    """
    When <season_frac> of the season is completed, this returns the fraction of the prediction to take from preseason
    Derived empirically by maximizing performance at a variety of points in the season and fitting a power law
    """
    return max(0, 1-season_frac)**2.6


def process_ap_sql(df_ap):
    """ Convert single SQL strings into lists of strings """

    for i in range(1, df_ap.shape[1]):
        df_ap[f'r{i}'] = df_ap[f'r{i}'].apply(lambda x:[y.strip() for y in x.strip('{}').split(',')])
