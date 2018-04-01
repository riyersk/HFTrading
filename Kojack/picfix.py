#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 10:54:54 2018

@author: subramanianiyer
"""

import pickle
import os
import sys
with open('check3x1.pkl', 'rb') as picklefile:
    data = pickle.load(picklefile)
os.remove('check3x1.pkl')
with open('check3x1.pkl', 'wb') as picklefile:
    pickle.dump(data, picklefile, protocol=2)