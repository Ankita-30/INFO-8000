# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:34:27 2020

@author: ar54482
"""

import requests #great web library
import pandas as pd #great table/dataframe library
import csv

doc_id = "1lSr2lJab-cQpWBaT4ZB1AWxqLa0QOFhsmtGZ3vehzqE" #from your shared link
url = f"https://docs.google.com/spreadsheets/d/{doc_id}/gviz/tq?tqx=out:csv" #https://docs.google.com/spreadsheets/d/1lSr2lJab-cQpWBaT4ZB1AWxqLa0QOFhsmtGZ3vehzqE/edit?usp=sharing
res = requests.get(url,allow_redirects=True)
open("temp.csv",'wb').write(res.content) #temporarily write to a file
df = pd.read_csv('temp.csv') #so that we can read it. 
print(df.head()) #so we can see that it worked

