import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

shares = pd.read_pickle('errors_share.pkl')
bp = pd.read_pickle('bin_prices.pkl')
errors = pd.read_pickle('errors_main.pkl')

T_SUMMARY = [0, 3, 6, 9, 12, 15]
SM = ['M2', 'M3', 'M4', 'M5', 'M6']
BIN_LABELS = [f'{(540 + i * 15) // 60:02d}:{(540 + i * 15) % 60:02d}' for i in range(18)]
BIN_IDX = {b: i for i, b in enumerate(BIN_LABELS)}

shares = shares.reset_index(drop=True)
shares['bin_idx'] = shares['bin_t'].map(BIN_IDX)
shares['abs_e'] = shares['e_share'].abs()

# --- Output 1: L1 share error table ---
l1 = (shares.groupby(['model', 'T', 'session_date'])['abs_e']
      .sum().reset_index(name='l1'))
l1_tbl = pd.DataFrame(
    {t: {m: l1[(l1['model'] == m) & (l1['T'] == t)]['l1'].mean()
         for m in SM}
     for t in T_SUMMARY}
)
l1_tbl.index.name = 'Model'
l1_tbl.columns.name = 'T'
print('=== Output 1: Mean Σ|e_share| (bin count varies by T — not comparable across T) ===')
print(l1_tbl.round(4).to_string())
print()

# --- Output 2: peak-bin accuracy ---
idx_true = shares.groupby(['model', 'T', 'session_date'])['s_true'].idxmax()
idx_hat = shares.groupby(['model', 'T', 'session_date'])['s_hat'].idxmax()
pk = pd.DataFrame({
    'true_peak': shares.loc[idx_true.values, 'bin_idx'].values,
    'hat_peak': shares.loc[idx_hat.values, 'bin_idx'].values,
}, index=idx_true.index).reset_index()
pk['hit'] = (pk['true_peak'] == pk['hat_peak']).astype(int)
pk['near'] = ((pk['true_peak'] - pk['hat_peak']).abs() <= 1).astype(int)
peak_tbl = pd.DataFrame(
    {t: {m: (f'{pk[(pk["model"]==m)&(pk["T"]==t)]["hit"].mean()*100:.0f}/'
             f'{pk[(pk["model"]==m)&(pk["T"]==t)]["near"].mean()*100:.0f}')
         for m in SM}
     for t in T_SUMMARY}
)
peak_tbl.index.name = 'Model'
peak_tbl.columns.name = 'T'
print('=== Output 2: Peak-bin accuracy hit%/near-hit% ===')
print(peak_tbl.to_string())
print()

# --- Output 3: share heatmap M2 ---
m2 = shares[shares['model'] == 'M2']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, t in enumerate(T_SUMMARY):
    ax = axes[i]
    gb = m2[m2['T'] == t].groupby('bin_t')['e_share']
    g_mean = gb.mean().reindex(BIN_LABELS[t:])
    g_sem = (gb.std() / gb.count().pow(0.5)).reindex(BIN_LABELS[t:])
    x = list(range(len(g_mean)))
    ax.bar(x, g_mean, yerr=g_sem, color='steelblue', alpha=0.7, capsize=3, ecolor='navy')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS[t:], rotation=90, fontsize=7)
    ax.set_title(f'T={t}', fontsize=9)
    ax.set_ylabel('Mean e_share ±1SE')
fig.suptitle('Share Errors M2 — Mean per Bin', fontsize=11)
fig.tight_layout()
fig.savefig('share_heatmap_m2.png', dpi=150)
plt.close(fig)

# --- Output 4: P&L table ---
sp = shares.merge(bp[['session_date', 'bin_t', 'P_vwap', 'VWAP_session']], on=['session_date', 'bin_t'])
sp['vwap_contrib'] = sp['e_share'] * sp['P_vwap']
vwe = (sp.groupby(['model', 'T', 'session_date'])
       .agg(VWAP_vol_err=('vwap_contrib', 'sum'), VWAP_session=('VWAP_session', 'first'))
       .reset_index())
vrem = errors[['model', 'T', 'session_date', 'V_rem_true']].copy()
vwe = vwe.merge(vrem, on=['model', 'T', 'session_date'])
vwe = vwe[vwe['V_rem_true'] >= 1e6].copy()
vwe['PnL_USD'] = 100e6 * vwe['VWAP_vol_err'].abs() / vwe['VWAP_session']
pnl_filt = vwe[vwe['T'].isin(T_SUMMARY) & (vwe['T'] >= 1)]
pnl_tbl = pd.DataFrame(
    {t: {m: pnl_filt[(pnl_filt['model'] == m) & (pnl_filt['T'] == t)]['PnL_USD'].mean()
         for m in SM}
     for t in T_SUMMARY if t >= 1}
)
pnl_tbl.index.name = 'Model'
pnl_tbl.columns.name = 'T'
print('=== Output 4: Mean PnL_USD (100mm position, T≥1) ===')
print(pnl_tbl.round(0).to_string())
print()

# --- Output 5: P&L histogram M2 T=6 ---
m2_t6 = vwe[(vwe['model'] == 'M2') & (vwe['T'] == 6)]
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(m2_t6['PnL_USD'], bins=25, color='steelblue', alpha=0.7, edgecolor='white')
for ref, lbl in [(1000, '$1k'), (5000, '$5k'), (10000, '$10k'), (25000, '$25k')]:
    ax.axvline(ref, color='red', ls='--', lw=1.2, label=lbl)
ax.set_xlabel('PnL_USD ($)')
ax.set_ylabel('Count')
ax.set_title('P&L Distribution — M2 T=6 (100mm position)')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig('pnl_hist_m2_t6.png', dpi=150)
plt.close(fig)

print('Figures saved: share_heatmap_m2.png  pnl_hist_m2_t6.png')
