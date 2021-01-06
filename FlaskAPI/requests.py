# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 04:46:50 2021

@author: abdo
"""

import requests

from data_input import data_in




URL = 'http://127.0.0.1:5000/predict'

headers = {"Content-Type" : "application/json"}

data = {"input": data_in}

r = requests.get(url = URL, headers = headers, json= data)

r.json()

