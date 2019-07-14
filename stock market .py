#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:10:27 2019

@author: yash_j1301
"""

import pandas_datareader as web
import datetime 

start=datetime.datetime(2019,6,1)
end=datetime.datetime(2019,7,14)

yes=web.DataReader('YESBANK.BO','yahoo', start, end)
yes

yes.columns
yes[['Open','Close', 'High','Low']].plot(figsize=(12,10))
