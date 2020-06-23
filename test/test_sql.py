import os
import unittest

import pandas as pd

from watchcbb.sql import SQLEngine

class TestSql(unittest.TestCase):

    fname = None

    @classmethod
    def setUpClass(cls):
        cls.fname = os.path.join(os.path.dirname(__file__), 'temp.txt')
        with open(cls.fname, 'w') as fid:
            fid.write('username\npassword\n')

    def test_SQLEngine_init(self):
        self.assertRaises(SQLEngine.SQLException, SQLEngine, 'cbb', 'kdahga.ahgaoig')
        self.assertRaises(SQLEngine.SQLException, SQLEngine, 'cbb', self.fname)
        self.assertRaises(SQLEngine.SQLException, SQLEngine, 'cbb', 1)
        self.assertRaises(SQLEngine.SQLException, SQLEngine, 'cbb', ['x'])
        

if __name__=="__main__":
    unittest.main()
