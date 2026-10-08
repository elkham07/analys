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

neg = d[d['diff'] < -0.01].copy()
neg['k327_flag'] = neg['k327']
neg['reset_to_0'] = neg['pokazanie'] < 1
s = ev[ev.sobytie == 'сброс'][['anon_id', 'data']].rename(columns={'data': 'd_ev'})
m = neg.merge(s, on='anon_id', how='left')
m['near'] = (m['data'] - m['d_ev']).abs().dt.days <= 2
neg['near_reset_event'] = neg.index.map(m.groupby(m.index)['near'].any()).fillna(False) if False else False
nr = m.groupby(['anon_id', 'data'])['near'].any()
neg['near_reset_event'] = [bool(nr.get((a, b), False)) for a, b in zip(neg.anon_id, neg.data)]
print('\n=== 1. ОТРИЦАТЕЛЬНЫЙ РАСХОД (diff < -0.01 м3; |-0.01..0| = шум float, отброшен)')
print('событий:', len(neg), ' приборов:', neg.anon_id.nunique())
print('  из них кратно 327.68 (сбой разряда):', int(neg.k327_flag.sum()), 'событий /', neg[neg.k327_flag].anon_id.nunique(), 'приб.')
print('  откат после всплеска (возврат):', int(neg.glitch_down.sum()), 'событий /', neg[neg.glitch_down].anon_id.nunique(), 'приб.')
print('  сброс к ~0 (<1 м3):', int(neg.reset_to_0.sum()), 'событий /', neg[neg.reset_to_0].anon_id.nunique(), 'приб.')
print('  рядом (±2 дн) с событием "сброс":', int(neg.near_reset_event.sum()), 'событий /', neg[neg.near_reset_event].anon_id.nunique(), 'приб.')
small = neg[(neg['diff'] > -1)]
print('  мелкий откат (-1..-0.01 м3):', len(small), 'событий /', small.anon_id.nunique(), 'приб.')
cnt = neg.groupby('anon_id').size().sort_values(ascending=False)
print('Топ по числу отрицательных событий:\n', cnt.head(10).to_string())
print('Топ по глубине падения:\n', neg.nsmallest(8, 'diff')[['anon_id', 'data', 'diff']].to_string(index=False))
neg.to_csv('out_1_negative.csv', index=False)
 
ok = d[(d['gap'] == 1) & (d['diff'] >= 0) & ~d['bad']].copy()   # чистые суточные приращения
med = ok[ok['diff'] > EPS].groupby('anon_id')['diff'].median().rename('med_nz')
ok = ok.merge(med, on='anon_id', how='left')
ABS = 5.0; REL = 10
ok['spike_abs'] = ok['diff'] > ABS
ok['spike_rel'] = (ok['diff'] > REL * ok['med_nz']) & (ok['diff'] > 1.0)   # >=1 м3 чтобы не ловить мелочь
sp = ok[ok.spike_abs | ok.spike_rel]
print('\n=== 2. ВСПЛЕСКИ (чистые суточные приращения, без сбоев телеметрии)')
print(f'абсолютный порог >{ABS} м3/сут: дней {ok.spike_abs.sum()}, приборов {ok[ok.spike_abs].anon_id.nunique()}')
print(f'относительный порог > {REL}x медианы ненулевого дня (и >1 м3): дней {ok.spike_rel.sum()}, приборов {ok[ok.spike_rel].anon_id.nunique()}')
print('  оба условия:', ok[ok.spike_abs & ok.spike_rel].anon_id.nunique(), 'приб.')
print('Топ:\n', sp.sort_values('diff', ascending=False).drop_duplicates('anon_id').head(10)[['anon_id', 'data', 'diff', 'med_nz', 'paketov_za_sutki']].to_string(index=False))
sp.to_csv('out_2_spikes.csv', index=False)

real = d[(d['diff'] > 0)]
unr_lim = real[real['rate'] > real['lim']]                      # выше физического предела прибора
unr_50 = real[real['rate'] > 50]
print('\n=== 3. НЕРЕАЛЬНЫЕ (>предела DN15 ~75 / DN20 ~120 м3/сут; плюс аномальные показания)')
print('расход выше предела:', len(unr_lim), 'событий /', unr_lim.anon_id.nunique(), 'приб.')
print('расход >50 м3/сут:', len(unr_50), 'событий /', unr_50.anon_id.nunique(), 'приб.')
print('>1000 м3/сут:', (real.rate > 1000).sum(), ' приборов:', real[real.rate > 1000].anon_id.nunique())
mxr = g['pokazanie'].max()
print('Показание счётчика >10000 м3:', (mxr > 10000).sum(), 'приб ->', mxr[mxr > 10000].round(0).to_dict())
print('Топ:\n', unr_lim.sort_values('rate', ascending=False).drop_duplicates('anon_id').head(10)[['anon_id', 'data', 'diff', 'pokazanie']].to_string(index=False))
unr_lim.to_csv('out_3_unreal.csv', index=False)


def best_run(vals):
    b = c = 0
    for v in vals:
        c = c + 1 if v else 0
        b = max(b, c)
    return b
 
leak = {}; zero = {}; zero_tail = {}
for a, x in d.groupby('anon_id'):
    df = x['diff'].values; gp = x['gap'].values; bd = x['bad'].values
    # непрерывное потребление: подряд суточные приращения >0 (gap==1)
    leak[a] = best_run((gp == 1) & (df > EPS)) # type: ignore
    # нули: показание не менялось подряд (любой gap, длительность считаем в днях)
    b = c = 0; dl = x['data'].values; start = None; bestd = 0
    for i in range(1, len(x)):
        if abs(df[i]) < EPS:
            if start is None: start = dl[i - 1]
            bestd = max(bestd, int((dl[i] - start) / np.timedelta64(1, 'D')))
        else:
            start = None
    zero[a] = bestd
    # ноль в хвосте периода (сейчас стоит)
    k = 0
    for i in range(len(x) - 1, 0, -1):
        if abs(df[i]) < EPS: k = int((dl[-1] - dl[i - 1]) / np.timedelta64(1, 'D'))
        else: break
    zero_tail[a] = k
leak = pd.Series(leak); zero = pd.Series(zero); zero_tail = pd.Series(zero_tail)
 

 