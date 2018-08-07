#! /usr/bin/python

import pandas as pd
import os

asanas_df = pd.read_csv('asanas_rem50.csv')

for asana in asanas_df.Sanskrit:
    asana = asana.replace(' ','+')
    print asana
    cmd = 'image_search google %s --limit 1000 --json' % asana
    os.system(cmd)