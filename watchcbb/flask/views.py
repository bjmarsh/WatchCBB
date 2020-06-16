import datetime as dt
import pickle
import gzip

from flask import Flask, g, render_template, request, Markup
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from matplotlib import cm

import watchcbb.teams
import watchcbb.utils as utils
from watchcbb.sql import SQLEngine
from watchcbb.flask import app

#######################################################################
# Load some constant global data that will be used in request function
#######################################################################

# connect to 'cbb' postgresql database. Uses credentials in watchcbb/PSQL_CREDENTIALS.txt
sql = SQLEngine('cbb')

# load dict of team data
teams = watchcbb.teams.teams_from_df(sql.df_from_query(""" SELECT * from teams """))

# Load database of historical AP rankings
df_ap = sql.df_from_query(""" SELECT * from ap_rankings """)
df_ap = df_ap.sort_values('date').reset_index(drop=True)
for i in range(1,26):
   df_ap[f'r{i}'] = df_ap[f'r{i}'].apply(lambda x:[y.strip() for y in x.strip('{}').split(',')])

# Get a map of season completion fraction
dates = sql.df_from_query(""" SELECT "Date" FROM game_data WHERE "Season"=2020 ORDER BY "Date" """).Date
TOTGAMES = dates.size
season_frac_dict = {date:(dates<date).mean() for date in dates}

# conference name mapping
conf_names = {
   "B10"   : "Big Ten",
   "B12"   : "Big 12",
   "ACC"   : "ACC",
   "SEC"   : "SEC",
   "BE"    : "Big East",
   "MWC"   : "MWC",
   "Amer"  : "American",
   "other" : "other",
}

def get_game_models(fname='models/game_regressions.pkl'):
   """ Load pre-trained models from pickle file """

   if 'game_models' not in g:
      with open(fname, 'rb') as fid:
         game_models = pickle.load(fid)
   return game_models

def get_reddit_model(fname='models/reddit_regression.pkl'):
   """ Load pre-trained reddit model from pickle file """

   if 'reddit_model' not in g:
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


###############################################################################################
#
# App routing functions:
# - there is only one page, / or /index
# - The route /games performs the request in the background and returns an HTML table snippet
# - index.html is dynamically updated with javascript to reflect returned table
#
###############################################################################################


@app.route('/')
@app.route('/index')
def index():
   return render_template("index.html",
       title = 'WatchCBB'
   )


@app.route('/games', methods=['GET'])
def get_games():
   """ Main request that returns a list of recommended games """

   # get the input that the client sends
   date = request.args.get('date', "")
   # three slider values
   s1 = float(request.args.get('s1', 0))
   s2 = float(request.args.get('s2', 0))
   s3 = float(request.args.get('s3', 0))
   checks = request.args.get('checks', '11111111')

   allowed_confs = []
   for i,c in enumerate(["B10","B12","ACC","SEC","BE","MWC","Amer","other"]):
      if checks[i]=='1':
         allowed_confs.append(conf_names[c])

   try:
      date = dt.date(*[int(x) for x in date.split('-')])
   except:
      return """<big><p style="color:red;">Please enter a date in the format YYYY-mm-dd</p></big>"""

   date_end = date+dt.timedelta(7)

   # get current AP rankings
   idx = np.argmax(date < df_ap.date) - 1
   ranks = df_ap.iloc[idx].values[1:].tolist()


   sql_query = """
       SELECT * FROM game_data WHERE "Date">='{date}' AND "Date"<'{date_end}' ORDER BY "Date";
   """.format(date=date, date_end=date_end)

   df_games = sql.df_from_query(sql_query)

   # Update ranks to reflect rank right now, not at time of game
   df_games.Wrank = df_games.WTeamID.apply(lambda tid: get_rank(ranks, tid)).fillna(-1)
   df_games.Lrank = df_games.LTeamID.apply(lambda tid: get_rank(ranks, tid)).fillna(-1)
   
   try:
      with gzip.open("data/season_stats/2020/{0}.pkl.gz".format(date), 'rb') as fid:
         season_stats_dict, season_stats_df = pickle.load(fid)
   except:
      return """<big><p style="color:red;">Invalid date! Must be during the 2019-20 season.</p></big>"""

   season_frac = season_frac_dict[date]

   # get data into format that can be fed into models
   data = utils.compile_training_data(df_games, season_stats_dict, sort='alphabetical', include_preseason=True)


   ## get mean/std of pace for use later
   mean_pace = season_stats_df.CompositePace.mean()
   std_pace = season_stats_df.CompositePace.std()

   ## Load game and reddit models from pickle files
   pca, logreg, logreg_simple, linreg_pace, linreg_margin, linreg_total = get_game_models()
   linreg_reddit = get_reddit_model()

   ## adjust coefficients from sliders (which are in the range [-100,100])
   ## coefficients are NetEffSum, Upset prob, |pred_margin|, is_rivalry
   linreg_reddit.coef_[0] -= 0.02*s1/100
   linreg_reddit.coef_[2] -= 0.02*s1/100
   linreg_reddit.coef_[1] *= ((s2+100)/100)**2.5
   linreg_reddit.coef_ = np.append(linreg_reddit.coef_, [0.0])
   linreg_reddit.coef_[3] /= 3
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

   print(mean_pace, std_pace)
   print(data[["gid",'pace1','pace2',"prob","compratsum","pred_pace","pred_margin",
               "upset_prob","is_rivalry","reddit_score","preseason_paceprod"]].head(20))

   games = []
   nselected = 0
   for i,row in data.iterrows():
      if nselected >= 10:
         break
      
      datestr = "{0}, {1}/{2}".format(row.date.strftime("%a"), row.date.month, row.date.day)
      t1, t2 = row.tid1, row.tid2

      c1, c2 = teams[t1].conference, teams[t2].conference
      if c1 not in conf_names.values():
         c1 = 'other'
      if c2 not in conf_names.values():
         c2 = 'other'
      if c1 not in allowed_confs and c2 not in allowed_confs:
         continue

      vs_str = '@'
      if row.HA == 1:
         t1, t2 = t2, t1
      if row.HA == 0:
         vs_str = 'vs.'
      r1 = get_rank(ranks, t1)
      r2 = get_rank(ranks, t2)
      r1str = Markup(f'<small>({r1})</small>' if r1 else '')
      r2str = Markup(f'<small>({r2})</small>' if r2 else '')
      
      fmt_upset_prob = "{0:d}%".format(int(round(row.upset_prob*100)))
      if fmt_upset_prob=="0%":
         fmt_upset_prob = ""
      else:
         rgba = cm.get_cmap('Reds')(row.upset_prob*0.75 + 0.25)
         c = rgba2hex(rgba)
         fmt_upset_prob = Markup(f"""<b><p style="color:{c};">{fmt_upset_prob}</p></b>""")
         
      fmt_margin = "{:+d}".format(int(round(row.pred_margin)))

      pace_string = ""
      if row.pred_pace > 2.0:
         pace_string = "<p style='color:{};'>Very fast</p>".format(rgba2hex(cm.get_cmap('Greens')(0.9)))
      elif row.pred_pace > 1.0:
         pace_string = "<p style='color:{};'>Fast</p>".format(rgba2hex(cm.get_cmap('Greens')(0.5)))
      elif row.pred_pace < -2.0:
         pace_string = "<p style='color:{};'>Very slow</p>".format(rgba2hex(cm.get_cmap('Reds')(0.9)))
      elif row.pred_pace < -1.0:
         pace_string = "<p style='color:{};'>Slow</p>".format(rgba2hex(cm.get_cmap('Reds')(0.5)))
      pace_string = Markup(pace_string)

      is_rivalry_text = "" if not row.is_rivalry else "Rivalry!"
         
      nselected += 1
      games.append(dict(
         i = i+1,
         date = datestr, 
         t1 = t1,
         t2 = t2,
         dn1 = teams[t1].display_name,
         dn2 = teams[t2].display_name,
         vs_str = vs_str,
         r1str = r1str,
         r2str = r2str,
         is_rivalry_text = is_rivalry_text,
         upset_prob = fmt_upset_prob,
         margin = fmt_margin,
         pace_string = pace_string
      ))

   return render_template('games_table.html',games=games)
