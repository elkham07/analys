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
 

def leak_flat(x):
    df = x['diff'].values; gp = x['gap'].values; best = 0; cur = []
    for v, q in zip(df, gp):
        if q == 1 and v > EPS: cur.append(v)
        else:
            if len(cur) >= 30:
                a = np.array(cur)
                if a.std() / a.mean() < 0.5 and a.min() > 0.01: best = max(best, len(cur))
            cur = []
    if len(cur) >= 30:
        a = np.array(cur)
        if a.std() / a.mean() < 0.5 and a.min() > 0.01: best = max(best, len(cur))
    return best
leak_f = d[~d.bad].groupby('anon_id').apply(leak_flat)
print('\n=== 4. УТЕЧКА (непрерывное потребление без нулевых суток)')
for t in (14, 30, 60, 90): print(f'  серия >= {t} суток подряд без нуля: {(leak >= t).sum()} приб.') # type: ignore
print('  Критерий итоговый: >=30 подряд И ровный профиль (CV<0.5, min>0.01):', (leak_f >= 30).sum(), 'приб.')
print('Топ (длина серии):', leak_f.sort_values(ascending=False).head(10).to_dict())
leak_f[leak_f >= 30].to_csv('out_4_leak.csv', header=['days'])
 
print('\n=== 5. НУЛЕВОЕ потребление (показание не меняется, пакеты идут)')
for t in (30, 60, 90, 180): print(f'  серия >= {t} суток: {(zero >= t).sum()} приб.') # type: ignore # type: ignore
print('  из них нули держатся до конца периода (>=30 сут в хвосте):', (zero_tail >= 30).sum(), 'приб.') # type: ignore
print('Топ:', zero.sort_values(ascending=False).head(10).to_dict())
pk = d.groupby('anon_id')['paketov_za_sutki'].mean()
z30 = zero[zero >= 30].index # type: ignore
print('  среднее пакетов/сут у "нулевых" >=30:', round(pk[z30].mean(), 2), ' у остальных:', round(pk.drop(z30).mean(), 2))
zero[zero >= 30].to_csv('out_5_zero.csv', header=['days']) # type: ignore
 

clean = d[(d['diff'] >= 0) & ~d['bad'] & (d['gap'] == 1)]
step = []
for a, x in clean.groupby('anon_id'):
    if len(x) < 120: continue
    h = len(x) // 2
    f, s2 = x.iloc[:h]['diff'], x.iloc[h:]['diff']
    mf, ms = f.mean(), s2.mean()
    if max(mf, ms) < 0.3: continue
    r = (ms + 1e-9) / (mf + 1e-9)
    mdf, mds = f.median(), s2.median()
    if r >= 3 and mds > 2 * max(mdf, 0.01) or r <= 1 / 3 and mdf > 2 * max(mds, 0.01):
        step.append((a, round(mf, 3), round(ms, 3), round(r, 2), x.iloc[h]['data'].date()))
step = pd.DataFrame(step, columns=['anon_id', 'mean_1half', 'mean_2half', 'ratio', 'split_date'])
print('\n=== 6. СТУПЕНЬКА (среднесут. расход 2-й половины / 1-й; >=3x или <=1/3, оба по среднему и медиане, >=120 чистых дней)')
print('вверх:', (step.ratio >= 3).sum(), ' вниз:', (step.ratio <= 1/3).sum())
print(step.sort_values('ratio', ascending=False).head(8).to_string(index=False))
print(step.sort_values('ratio').head(5).to_string(index=False))
step.to_csv('out_6_step.csv', index=False)
 

mg = ev[ev.sobytie == 'магнит']
mag_dev = set(mg.anon_id)
print('\n=== 7. СВЯЗЬ С МАГНИТОМ')
print('приборов с магнит-событиями:', len(mag_dev), 'из 3000 (', round(len(mag_dev) / 30, 1), '%)')
flags = pd.DataFrame(index=zero.index)
flags['magnet'] = flags.index.isin(mag_dev)
flags['zero30'] = zero >= 30 # type: ignore
flags['neg'] = flags.index.isin(neg.anon_id)
flags['spike'] = flags.index.isin(sp.anon_id)
flags['unreal'] = flags.index.isin(unr_lim.anon_id)
flags['leak'] = leak_f.reindex(flags.index).fillna(0) >= 30
flags['step'] = flags.index.isin(step.anon_id)
flags['step_down'] = flags.index.isin(step[step.ratio <= 1/3].anon_id)
flags['step_up'] = flags.index.isin(step[step.ratio >= 3].anon_id)
print('доля приборов с магнитом в группе (vs всего %.1f%%):' % (flags.magnet.mean() * 100))
for c in ['zero30', 'neg', 'spike', 'unreal', 'leak', 'step', 'step_down', 'step_up']:
    n = flags[c].sum(); w = flags[flags[c]].magnet.sum()
    base = flags[~flags[c]].magnet.mean()
    print(f'  {c:9s}: n={n:4d}, с магнитом {w:4d} ({100*w/max(n,1):.1f}%), без аномалии {100*base:.1f}%')


daily = d[(d['gap'] == 1) & (d['diff'] >= 0) & ~d['bad']].set_index(['anon_id', 'data'])['diff']
rows = []
first = mg.groupby('anon_id')['data'].min()
for a, t in first.items():
    if a not in daily.index.get_level_values(0): continue
    s = daily.loc[a] # type: ignore
    b = s[(s.index >= t - pd.Timedelta(days=14)) & (s.index < t)]
    af = s[(s.index > t) & (s.index <= t + pd.Timedelta(days=14))]
    if len(b) >= 7 and len(af) >= 7: rows.append((a, b.mean(), af.mean()))
es = pd.DataFrame(rows, columns=['anon_id', 'before', 'after'])
es = es[es.before > 0.01]
es['ratio'] = es.after / es.before
print(f'\nЕvent-study (14 дн до vs 14 дн после ПЕРВОГО магнита), приборов {len(es)}:')
print('  медиана отношения после/до =', round(es.ratio.median(), 2), ' доля с падением >50%:', round((es.ratio < 0.5).mean() * 100, 1), '%  с ростом >2x:', round((es.ratio > 2).mean() * 100, 1), '%')
# плацебо: случайная дата в активный период тех же приборов
rng = np.random.default_rng(0); rows = []
for a in es.anon_id:
    s = daily.loc[a]
    if len(s) < 60: continue
    t = s.index[rng.integers(20, len(s) - 20)]
    b = s[(s.index >= t - pd.Timedelta(days=14)) & (s.index < t)]
    af = s[(s.index > t) & (s.index <= t + pd.Timedelta(days=14))]
    if len(b) >= 7 and len(af) >= 7 and b.mean() > 0.01: rows.append(af.mean() / b.mean())
pl = pd.Series(rows)
print('  плацебо (случайная дата): медиана', round(pl.median(), 2), ' с падением >50%:', round((pl < 0.5).mean() * 100, 1), '%  с ростом >2x:', round((pl > 2).mean() * 100, 1), '%')
# магнит по типам приборов
pr2 = pr.set_index('anon_id')
fl = flags.join(pr2[['tip', 'bs', 'model']])
print('\nДоля приборов с магнитом по типу:'); print(fl.groupby('tip').magnet.agg(['mean', 'size']).round(3).to_string())
print('по BS:'); print(fl.groupby('bs').magnet.mean().round(3).to_string())
flags.to_csv('out_7_flags.csv')

