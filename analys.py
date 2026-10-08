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


d['next_diff'] = g['diff'].shift(-1)
# всплеск, который на следующем отсчёте почти полностью откатился назад
d['glitch_up'] = (d['diff'] > 20) & (d['next_diff'] < -0.8 * d['diff'])
d['glitch_down'] = d['glitch_up'].shift(1, fill_value=False) & (d['diff'] < 0)
# скачок, кратный 327.68 (старший бит счётчика)
kk = (d['diff'] / K)
d['k327'] = ((kk - kk.round()).abs() * K < 1.0) & (kk.round() != 0) & (d['diff'].abs() > 100)
d['bad'] = d['glitch_up'] | d['glitch_down'] | d['k327'] | (d['diff'] < -EPS) | (d['diff'] > d['lim'] * d['gap'])
 
res = {}
def top(df, col, n=10):
    return df.sort_values(col, ascending=False).head(n)

