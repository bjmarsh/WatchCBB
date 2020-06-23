import os
import unittest

from sqlalchemy import Column, String, Integer, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import pandas as pd

from watchcbb.sql import SQLEngine

Base = declarative_base()

class TeamsTable(Base):
    __tablename__ = "teams"

    team_id = Column(String, primary_key=True)
    conference = Column(String, nullable=True)
    year_start = Column(Integer, nullable=False)
    year_end = Column(Integer, nullable=False)

    def __init__(self, team_id, conference, year_start, year_end):
        self.team_id = team_id
        self.conference = conference
        self.year_start = year_start
        self.year_end = year_end

class TestSql(unittest.TestCase):

    fname = None

    sql = SQLEngine('blah', credentials="MEMORY")
    Session = sessionmaker(bind=sql.engine)
    session = Session()

    teams_data = [
        ['purdue', 'Big Ten', 1950, 2020],
        ['kansas', 'Big 12', 1970, 2020],
        ['fakeschool', 'ABC', 1980, 2010],
    ]

    @classmethod
    def setUpClass(cls):
        cls.fname = os.path.join(os.path.dirname(__file__), '.temp.txt')
        with open(cls.fname, 'w') as fid:
            fid.write('username\npassword\n')

        Base.metadata.create_all(cls.sql.engine)
        for row in cls.teams_data:
            cls.session.add(TeamsTable(*row))
        cls.session.commit()

    def test_SQLEngine_init(self):
        """ check that bad inputs all raise the correct error """
        self.assertRaises(SQLEngine.SQLException, SQLEngine, 'cbb', None, testing=True)
        self.assertRaises(SQLEngine.SQLException, SQLEngine, 'cbb', 'kdahga.ahgaoig') # non-existent file
        self.assertRaises(SQLEngine.SQLException, SQLEngine, 'cbb', self.fname)  # invalid credentials
        self.assertRaises(SQLEngine.SQLException, SQLEngine, 'cbb', 1)
        self.assertRaises(SQLEngine.SQLException, SQLEngine, 'cbb', ['x'])

    def test_SQLEngine_get_teams_active_in_year(self):
        """ test a variety of years to get all edge cases """

        for year in [1940, 1960, 1970, 1975, 2000, 2015, 2030]:
            tids = self.sql.get_teams_active_in_year(year)
            self.assertEqual(tids, [x[0] for x in self.teams_data if x[2]+1<=year and x[3]>=year])
        
    def test_SQLEngine_df_from_query(self):
        """ test a couple db operations on existing 'teams' table """

        vals = self.sql.df_from_query(""" SELECT * FROM teams """).values.tolist()
        self.assertEqual(vals, self.teams_data)

        vals = self.sql.df_from_query(""" SELECT * FROM teams WHERE year_start<1960 """).values.tolist()
        self.assertEqual(vals, self.teams_data[:1])
        
    def test_SQLEngine_df_to_sql(self):
        """ create a dummy table to test SQL writing function """

        data = [[1,2],[3,4]]
        df = pd.DataFrame(data, columns=['a','b']) 
        self.sql.df_to_sql(df, "test_table", if_exists='replace')
        vals = self.sql.df_from_query(""" SELECT * FROM test_table """).values.tolist()

        self.assertEqual(vals, data)

    def test_SQLEngine_drop_rows(self):
        """ create a dummy table and test dropping a row """

        data = [[1,2],[3,4]]
        df = pd.DataFrame(data, columns=['a','b']) 
        self.sql.df_to_sql(df, "test_table", if_exists='replace')
        self.sql.drop_rows("test_table", "a==1")
        vals = self.sql.df_from_query(""" SELECT * FROM test_table """).values.tolist()

        self.assertEqual(vals, data[1:])


if __name__=="__main__":
    unittest.main()
