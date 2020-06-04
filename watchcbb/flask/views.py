from flask import Flask, render_template, request
import pandas as pd

import watchcbb.sql as sql
from watchcbb.flask import app

@app.route('/')
@app.route('/index')
def index():
   return render_template("index.html",
       title = 'WatchCBB'
   )

@app.route('/games', methods=['GET'])
def get_games():
   date = request.args.get('date')
   s1 = request.args.get('s1')
   s2 = request.args.get('s2')
   s3 = request.args.get('s3')

   # print(request.args)

   sql_query = """
       SELECT "Date","gid" FROM game_data WHERE "Date"='{date}';
   """.format(date=date)

   query_results = sql.df_from_query(sql_query)

   games = []
   for i in range(0,query_results.shape[0]):
       games.append(dict(
          date=str(query_results.Date[i]), 
          gid=query_results.gid[i],
          sliders="{0}_{1}_{2}".format(s1,s2,s3),
       ))
   return render_template('games_table.html',games=games)
