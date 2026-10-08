import pandas as pd, numpy as np

K = 327.68            
EPS = 0.0005          


rd = lambda f: pd.read_csv(f, sep=';', encoding='utf-8-sig')
d = rd('pokazaniya.csv'); d['data'] = pd.to_datetime(d['data'])
pr = rd('pribory.csv'); ev = rd('sobytiya.csv'); ev['data'] = pd.to_datetime(ev['data'])
ev = ev.drop_duplicates()                       

d = d.sort_values(['anon_id', 'data']).reset_index(drop=True)
g = d.groupby('anon_id')
d['gap'] = g['data'].diff().dt.days                 # дней между показаниями
d['diff'] = g['pokazanie'].diff().round(6)          # приращение накопительного счётчика
d['rate'] = d['diff'] / d['gap']   

lim = pr.set_index('anon_id')['model'].fillna('').map(lambda m: 120 if 'D20' in m else 75)
d['lim'] = d['anon_id'].map(lim)