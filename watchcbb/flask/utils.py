"""
Helper functions for main flask app
"""

import os
import datetime as dt
import pickle
import gzip

from flask import Markup

import numpy as np
import pandas as pd
from matplotlib import cm

from watchcbb.sql import SQLEngine
import watchcbb.utils as utils
import watchcbb.teams


def parse_date_string(datestr):
    """ Take string formatted like YYYY-mm-dd and return datetime.date object """
    return dt.date(*[int(x) for x in datestr.split('-')])

def load_season_stats(pickle_dir, date):
    """ load season stats dict/dataframe from gzipped pickle """
    with gzip.open(os.path.join(pickle_dir,"{0}.pkl.gz".format(date)), 'rb') as fid:
        season_stats_dict, season_stats_df = pickle.load(fid)
    return season_stats_dict, season_stats_df


def get_game_models(fname='models/game_regressions.pkl'):
    """ Load pre-trained models from pickle file """
    with open(fname, 'rb') as fid:
        game_models = pickle.load(fid)
    return game_models


def get_reddit_model(fname='models/reddit_regression.pkl'):
    """ Load pre-trained reddit model from pickle file """
    with open(fname, 'rb') as fid:
        reddit_model = pickle.load(fid)
    return reddit_model


def get_rank(ranks, tid):
    """ Find tid in a jagged array of ranks in a given week. Return None if unranked """
    for irank,ts in enumerate(ranks):
        if tid in ts:
            return irank+1
    return None


def rgba2hex(rgba):
    """ Convert rgba tuple (e.g. (0.5, 0.4, 0.2, 1.0)) to hex code """
    hx = ''
    for f in rgba[:3]:
        s = hex(int(f*255))[2:]
        if len(s)<2:
            s = '0'+s
        hx += s
    return "#"+hx


def get_current_ap_ranks(df_ap, date):
    """ from AP ranking dataframe get the list of rankings on a given date """
    idx = np.argmax(date < df_ap.date) - 1
    ranks = df_ap.iloc[idx].values[1:].tolist()
    return ranks
    

def get_games_between_dates(date, date_end, sql_engine):
    """ Get dataframe of all games in [date, date_end) (note exclusive of date_end) """

    sql_query = """
        SELECT * FROM game_data WHERE "Date">='{date}' AND "Date"<'{date_end}' ORDER BY "Date";
    """.format(date=date, date_end=date_end)
    
    return sql_engine.df_from_query(sql_query)


def make_predictions(data, s1, s2, s3, season_frac, mean_pace, std_pace):
    """
    Take in game data dataframe and add columns representing predictions of win prob, pace, margin, total score
    s1,s2,s3 are the slider values on a scale from -100 to 100
    season_frac is the fraction of the season completed so far (used for computing preseason blend parameter)
    mean_pace, std_pace are mean/std of team pace values, used to scale/normalize pace values
    """

    ## Load game and reddit models from pickle files
    pca, logreg, logreg_simple, linreg_pace, linreg_margin, linreg_total = get_game_models()
    linreg_reddit = get_reddit_model()
    
    ## adjust coefficients from sliders (which are in the range [-100,100])
    ## coefficients are NetEffSum, Upset prob, |pred_margin|, is_rivalry
    linreg_reddit.coef_[0] -= 0.022*s1/100
    linreg_reddit.coef_[2] -= 0.02*s1/100
    linreg_reddit.coef_[1] *= ((s2+100)/100)**2.5
    linreg_reddit.coef_[3] /= 3
    linreg_reddit.coef_ = np.append(linreg_reddit.coef_, [0.0])
    linreg_reddit.coef_[4] = s3/400

    xf = pca.transform(data[utils.ADVSTATFEATURES])
    for i in range(len(utils.ADVSTATFEATURES)):
        data["PCA"+str(i)] = xf[:,i]

     # the rows where at least one team has no game data
    bad_rows = ((data.effsum > 1000) | (data.pace1.isna()) | (data.pace2.isna()))

    # win probability
    probs_cur = np.clip(logreg.predict_proba(data[utils.PCAFEATURES + ['HA']])[:,1], 0.001, 0.999)
    probs_pre = logreg_simple.predict_proba(data[['preseason_effdiff','HA']])[:,1]
    p = (1-season_frac)**2.6
    probs_blend = 1 / (1 + np.exp(-p*(-np.log(1./probs_pre-1)) - (1-p)*(-np.log(1./probs_cur-1))))
    data["prob"] = probs_blend
    
    # pace
    pace_cur = linreg_pace.predict(np.array([data.pace1.fillna(0)*data.pace2.fillna(0)]).T)
    pace_pre = linreg_pace.predict(np.array([data.preseason_paceprod]).T)
    data["pred_pace"] = p*pace_pre + (1-p)*pace_cur
    
    margin_cur = linreg_margin.predict(np.array([pace_cur*data.effdiff, data.HA]).T)
    margin_pre = linreg_margin.predict(np.array([pace_pre*data.preseason_effdiff, data.HA]).T)
    data["pred_margin"] = p*margin_pre + (1-p)*margin_cur
    
    # for the bad rows, use pure preseason predictions
    data.loc[bad_rows, 'prob'] = probs_pre[bad_rows]
    data.loc[bad_rows, 'pred_pace'] = pace_pre[bad_rows]
    data.loc[bad_rows, 'pred_margin'] = margin_pre[bad_rows]
    
    data["pred_pace"] = (data["pred_pace"] - mean_pace) / std_pace
    data["abs_pred_margin"] = data["pred_margin"].abs()
    data["upset_prob"] = data.apply(utils.get_df_upset_prob, axis=1)
    data["is_rivalry"] = data.apply(utils.is_rivalry, axis=1).astype(int)
    
    data["reddit_score"] = 10**linreg_reddit.predict(
        np.array([data.compratsum, data.upset_prob**2, data.abs_pred_margin, data.is_rivalry, data.pred_pace]).T
    ) - 1
    
    data = data.sort_values('reddit_score', ascending=False).reset_index(drop=True)

    return data


def is_allowed_conference(c1, c2, conf_names, allowed_confs):
    """
    Return True if at least one of c1/c2 is an allowed_conference
    conf_names is a list of all defined conferences (used to group things into an "other" category)
    """
    if c1 not in conf_names:
        c1 = 'other'
    if c2 not in conf_names:
        c2 = 'other'
    if c1 not in allowed_confs and c2 not in allowed_confs:
        return False
    return True


def get_formatted_upset_prob(p):
    """ Take the upset probability as a float and return an HTML-styled/colored string representation """
    fmt_upset_prob = "{0:d}%".format(int(round(100*p)))
    if fmt_upset_prob=="0%":
        fmt_upset_prob = ""
    else:
        rgba = cm.get_cmap('Reds')(p*0.75 + 0.25)
        c = rgba2hex(rgba)
        fmt_upset_prob = Markup(f"""<b><p style="color:{c};">{fmt_upset_prob}</p></b>""")

    return fmt_upset_prob


def get_pace_string(pace):
    """ Take a pace-of-play float and return an HTML-styled/colored string representation """

    pace_string = ""
    if pace > 2.0:
        pace_string = "<p style='color:{};'>Very fast</p>".format(rgba2hex(cm.get_cmap('Greens')(0.9)))
    elif pace > 1.0:
        pace_string = "<p style='color:{};'>Fast</p>".format(rgba2hex(cm.get_cmap('Greens')(0.5)))
    elif pace < -2.0:
        pace_string = "<p style='color:{};'>Very slow</p>".format(rgba2hex(cm.get_cmap('Reds')(0.9)))
    elif pace < -1.0:
        pace_string = "<p style='color:{};'>Slow</p>".format(rgba2hex(cm.get_cmap('Reds')(0.5)))
    return Markup(pace_string)
    
