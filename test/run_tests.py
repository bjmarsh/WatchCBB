#! /usr/bin/env python

import os
import unittest

if __name__=="__main__":

    test_dir = os.path.dirname(__file__)
    test_patt = "*_t.py"
    tests = unittest.TestLoader().discover(test_dir, test_patt)
    test_runner = unittest.TextTestRunner()
    test_runner.run(tests)

