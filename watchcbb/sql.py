import os
from sqlalchemy import create_engine
import sqlalchemy.exc

import pandas as pd


class SQLEngine:
    def __init__(self, dbname, credentials=None):
        """ Open a connection to a postgresql database
            
            Parameters:
            - dbname: is the name of the database
            - credentials: either a tuple with (username,password), or a filename
                           with username and password on first two lines.
                           Defaults to {__file__}/PSQL_CREDENTIALS.txt
        """
        if credentials is None:
            credentials = os.path.join(os.path.dirname(__file__),"PSQL_CREDENTIALS.txt")

        if type(credentials) == str:
            if not os.path.exists(credentials):
                raise SQLEngine.SQLException(f"Credential {credentials} filename does not exist")
            with open(credentials) as fid:
                credentials = [x.strip() for x in fid.readlines()]

        if type(credentials) in [tuple, list]:
            if len(credentials) != 2 or type(credentials[0]) != str or type(credentials[1]) != str:
                raise SQLEngine.SQLException("credentials must be a (user,password) tuple or filename")
            username = credentials[0]
            password = credentials[1]
        else:
            raise SQLEngine.SQLException("credentials must be a (user,password) tuple or filename")

        self.engine = create_engine('postgresql://%s:%s@localhost/%s'%(credentials[0], credentials[1], dbname))
        try:
            self.engine.connect()
        except sqlalchemy.exc.OperationalError:
            raise SQLEngine.SQLException("invalid credentials or dbname")
        
    def df_from_query(self, query):
        return pd.read_sql_query(query, self.engine)

    def df_to_sql(self, df, table_name, if_exists='fail'):
        df.to_sql(table_name, self.engine, index=False, if_exists=if_exists)

    class SQLException(Exception):
        pass


if __name__=="__main__":
    sql = SQLEngine('cbb')
    print(sql.df_from_query(""" SELECT * FROM teams LIMIT 5 """))
