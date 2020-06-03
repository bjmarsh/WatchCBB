import os
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database

import pandas as pd


with open(os.path.join(os.path.dirname(__file__),"PSQL_CREDENTIALS.txt")) as fid:
    cred = [x.strip() for x in fid.readlines()]

dbname = 'cbb'
user = cred[0]
pswd = cred[1]

## 'engine' is a connection to a database
## Here, we're using postgres, but sqlalchemy can connect to other things too.
engine = create_engine('postgresql://%s:%s@localhost/%s'%(user,pswd,dbname))

def df_from_query(query):
    return pd.read_sql_query(query, engine)

