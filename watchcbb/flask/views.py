import datetime as dt
import pickle
import gzip

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from matplotlib import cm

import watchcbb.utils as utils
import watchcbb.sql as sql
from watchcbb.flask import app

# Get a dictionary of nice "display names"
df_teams = sql.df_from_query(""" SELECT * from teams """)
disp_names = dict(zip(df_teams.team_id.values, df_teams.display_name.values))

# Load database of historical AP rankings
df_ap = sql.df_from_query(""" SELECT * from ap_rankings """)
df_ap = df_ap.sort_values('date').reset_index(drop=True)
for i in range(1,26):
   df_ap[f'r{i}'] = df_ap[f'r{i}'].apply(lambda x:[y.strip() for y in x.strip('{}').split(',')])

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
   date = request.args.get('date')

   # three slider values
   s1 = float(request.args.get('s1'))
   s2 = float(request.args.get('s2'))
   s3 = float(request.args.get('s3'))

   try:
      date = dt.date(*[int(x) for x in date.split('-')])
   except:
      return """<big><p style="color:red;">Please enter a date in the format YYYY-mm-dd</p></big>"""

   date_end = date+dt.timedelta(7)

   sql_query = """
       SELECT * FROM game_data WHERE "Date">='{date}' AND "Date"<'{date_end}';
   """.format(date=date, date_end=date_end)

   df_games = sql.df_from_query(sql_query)
   
   try:
      with gzip.open("data/season_stats/{0}.pkl.gz".format(date), 'rb') as fid:
         season_stats_dict, season_stats_df = pickle.load(fid)
   except:
      return """<big><p style="color:red;">Invalid date! Must be during the 2019-20 season.</p></big>"""

   data = utils.compile_training_data(df_games, season_stats_dict, sort='alphabetical')

   ## get mean/std of pace for use later
   mean_pace = season_stats_df.pace.mean()
   std_pace = season_stats_df.pace.std()

   with open('models/game_regressions.pkl', 'rb') as fid:
      pca, logreg, linreg_pace, linreg_margin = pickle.load(fid)
   with open('models/reddit_regression.pkl', 'rb') as fid:
      linreg_reddit = pickle.load(fid)

   ### adjust coefficients from sliders
   linreg_reddit.coef_[1] *= (s2+100)/100
   linreg_reddit.coef_ = np.append(linreg_reddit.coef_, [0.0])
   linreg_reddit.coef_[4] = s3/200
   linreg_reddit.coef_[0] -= 0.02*s1/100
   linreg_reddit.coef_[2] -= 0.02*s1/100

   xf = pca.transform(data[utils.ADVSTATFEATURES])
   for i in range(len(utils.ADVSTATFEATURES)):
      data["PCA"+str(i)] = xf[:,i]

   probs = logreg.predict_proba(data[utils.PCAFEATURES + ['HA']])[:,1]
   data["prob"] = probs
   data["pred_pace"] = linreg_pace.predict(np.array([data.pace1*data.pace2]).T)
   data["pred_margin"] = linreg_margin.predict(
      np.array([data["pred_pace"]*data.effdiff, data.HA]).T
      )
   data["pred_pace"] = (data["pred_pace"] - mean_pace) / std_pace
   data["abs_pred_margin"] = data["pred_margin"].abs()
   data["upset_prob"] = data.apply(utils.get_df_upset_prob, axis=1)
   data["is_rivalry"] = data.apply(utils.is_rivalry, axis=1).astype(int)
   
   data["reddit_score"] = 10**linreg_reddit.predict(
      np.array([data.neteffsum, data.upset_prob, data.abs_pred_margin, data.is_rivalry, data.pred_pace]).T
   ) - 1

   data = data.sort_values('reddit_score', ascending=False).reset_index(drop=True)

   print(data[["gid","prob","neteffsum","pred_pace","pred_margin",
               "upset_prob","is_rivalry","reddit_score"]].head(20))

   idx = np.argmax(date < df_ap.date) - 1
   ranks = df_ap.iloc[idx].values[1:].tolist()

   games = []
   for i,row in data.iterrows():
      if i >= 10:
         break
      
      datestr = "{0}, {1}/{2}".format(row.date.strftime("%a"), row.date.month, row.date.day)
      t1, t2 = row.tid1, row.tid2
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
         
      games.append(dict(
         i = i+1,
         date = datestr, 
         t1 = t1,
         t2 = t2,
         dn1 = disp_names[t1],
         dn2 = disp_names[t2],
         vs_str = vs_str,
         r1str = r1str,
         r2str = r2str,
         upset_prob = fmt_upset_prob,
         margin = fmt_margin,
         pace_string = pace_string
      ))

   return render_template('games_table.html',games=games)
