import os
from sqlalchemy import create_engine
import sqlalchemy.exc

import pandas as pd

class SQLEngine:
    """Handle the connection to the main 'cbb' database, and perform various database operations"""

    def __init__(self, dbname, credentials=None, testing=False):
        """ Open a connection to a postgresql database
            
            Parameters:
            - dbname: is the name of the database
            - credentials: either a tuple with (username,password), or a filename
                           with username and password on first two lines.
                           Defaults to {__file__}/PSQL_CREDENTIALS.txt
        """

        # connect to an in-memory database used for unit testing
        if credentials == "MEMORY":
            self.engine = create_engine('sqlite:///:memory:')
            return

        if credentials is None:
            fname = "PSQL_CREDENTIALS.txt" if not testing else "fake.file"
            credentials = os.path.join(os.path.dirname(__file__), fname)

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

    def get_teams_active_in_year(self, year):
        return self.df_from_query(
            f""" SELECT team_id FROM teams WHERE year_start+1 <= {year} AND year_end >= {year} """
            ).values.flatten().tolist()
        
    def df_from_query(self, query):
        return pd.read_sql_query(query, self.engine)

    def df_to_sql(self, df, table_name, if_exists='fail'):
        df.to_sql(table_name, self.engine, index=False, if_exists=if_exists)

    def drop_rows(self, table_name, condition):
        result = self.engine.execute(f""" DELETE FROM {table_name} WHERE {condition} """)
        return result


    class SQLException(Exception):
        pass


if __name__=="__main__":
    sql = SQLEngine('cbb')
    print(sql.df_from_query(""" SELECT * FROM teams LIMIT 5 """))
