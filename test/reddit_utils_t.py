import os
import datetime as dt
import unittest

import pandas as pd

import watchcbb.reddit_utils as ru

class TestTeams(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass

    def test_date_from_timestamp(self):
        ts1 = 1592557200 # 2020-06-19, 09:00
        ts2 = 1592575200 # 2020-06-19, 14:00
        date1 = ru.date_from_timestamp(ts1)
        date2 = ru.date_from_timestamp(ts2)
        self.assertIsInstance(date1, dt.date)
        self.assertIsInstance(date2, dt.date)
        # even though date is same, function should count
        # times before 10:00 as previous day
        self.assertEqual((date2-date1).days, 1)
        
    def test_parse_title(self):
        f = ru.parse_title
        # this sould return None b/c there is no split word between Purdue/Syracuse
        self.assertEqual(f('[Game Thread] Purdue Syracuse, 11/20'), None)
        r = f("[PostgameThread] #14 pURDUE (blah) defeats New  Hampshir'e St. in 4OT, 99-88")
        self.assertIsInstance(r, list)
        self.assertEqual(r, ['purdue', 'new-hampshire-st.'])

    def test_fix_names(self):
        f = ru.fix_names
        self.assertEqual(f(None), None)
        self.assertEqual(f('purdue'), None)
        for i in range(1,6):
            self.assertEqual(f(['texas-a&m']*i), ['texas-am']*i)
        self.assertEqual(f(['purdue']), ['purdue'])
        self.assertEqual(f(['st.-bonaventure']), ['st-bonaventure'])
        self.assertEqual(f(['st.-peters']), ['saint-peters'])
        self.assertEqual(f(['uc-davis']), ['california-davis'])
        self.assertEqual(f(['uri']), ['rhode-island'])
        self.assertEqual(f(['purduee']), None)
        self.assertEqual(ru._BAD, ['purduee'])


if __name__=="__main__":
    unittest.main()
