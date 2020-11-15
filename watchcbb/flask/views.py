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

import watchcbb.teams
import watchcbb.utils as utils
from watchcbb.sql import SQLEngine
from watchcbb.flask import utils as futils
from watchcbb.flask import app

#######################################################################
# Load some constant global data that will be used in request function
#######################################################################

# connect to 'cbb' postgresql database. Uses credentials in watchcbb/PSQL_CREDENTIALS.txt
sql = SQLEngine('cbb')

# load dict of team data
teams = watchcbb.teams.teams_from_df(sql.df_from_query(""" SELECT * from teams """))

# Load database of historical AP rankings
df_ap = sql.df_from_query(""" SELECT * from ap_rankings ORDER BY date""")
utils.process_ap_sql(df_ap)

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

    # convert check boxes to a list of allowed conferences
    allowed_confs = []
    for i,c in enumerate(["B10","B12","ACC","SEC","BE","MWC","Amer","other"]):
        if checks[i]=='1':
            allowed_confs.append(conf_names[c])

    try:
        date = futils.parse_date_string(date)
    except:
        return """<big><p style="color:red;">Please enter a date in the format YYYY-mm-dd</p></big>"""

    date_end = date+dt.timedelta(7)

    # get current AP rankings
    ranks = futils.get_current_ap_ranks(df_ap, date)

    # get games in current week
    df_games = futils.get_games_between_dates(date, date+dt.timedelta(7), sql)

    # Update ranks to reflect rank right now, not at time of game
    df_games.Wrank = df_games.WTeamID.apply(lambda tid: futils.get_rank(ranks, tid)).fillna(-1)
    df_games.Lrank = df_games.LTeamID.apply(lambda tid: futils.get_rank(ranks, tid)).fillna(-1)

    # load season stats from gzipped pickles
    try:
        season_stats_dict, season_stats_df = futils.load_season_stats('data/season_stats/2020', date)
    except:
        return """<big><p style="color:red;">Invalid date! Must be during the 2019-20 season.</p></big>"""

    # fraction of the season that has been completed so far -- used for computing preseason blend parameter
    season_frac = season_frac_dict[date]

    # get data into format that can be fed into models
    data = utils.compile_training_data(df_games, season_stats_dict, sort='alphabetical', include_preseason=True)

    ## get mean/std of pace for use later
    mean_pace = season_stats_df.CompositePace.mean()
    std_pace = season_stats_df.CompositePace.std()


    data = futils.make_predictions(data, s1, s2, s3, season_frac, mean_pace, std_pace)

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

        if not futils.is_allowed_conference(teams[t1].conference, teams[t2].conference, conf_names.values(), allowed_confs):
            continue
        
        vs_str = '@'
        if row.HA == 1:
            t1, t2 = t2, t1
        if row.HA == 0:
            vs_str = 'vs.'
        r1 = futils.get_rank(ranks, t1)
        r2 = futils.get_rank(ranks, t2)
        r1str = Markup(f'<small>({r1})</small>' if r1 else '')
        r2str = Markup(f'<small>({r2})</small>' if r2 else '')

        fmt_upset_prob = futils.get_formatted_upset_prob(row.upset_prob)
        fmt_margin = "{:+d}".format(int(round(row.pred_margin)))
        pace_string = futils.get_pace_string(row.pred_pace)
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


@app.route('/team/<team_id>')
def team_page(team_id):
    date = request.args.get('date', '2020-02-15')
    try:
        date = futils.parse_date_string(date)
    except:
        date = dt.date(2020,2,15)

    try:
        season_stats_dict, season_stats_df = futils.load_season_stats('data/season_stats/2020', date)
    except:
        return """Invalid date! Must be during the 2019-20 season."""

    stats = season_stats_dict[2020][team_id]

    team = teams[team_id]
    return render_template('team_page.html',
                           date = date,
                           team_id = team_id,
                           team_name = team.display_name,
                           city = team.location,
                           conference = team.conference,
                           stats = stats,
                           # wins = stats['wins'],
                           # losses = stats['losses'],
                    )

@app.route('/slides')
def slides_page():
    return render_template('slides.html', title="WatchCBB")


@app.route('/cbb/preseason')
def preseason_page():
    year = request.args.get('year', '2021')
    try:
        year = int(year)
    except:
        year = 2021
