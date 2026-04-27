import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

errors = pd.read_pickle('errors_main.pkl')
T_SUMMARY = [0, 3, 6, 9, 12, 15]
MODELS = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6']
SM = ['M2', 'M3', 'M4', 'M5', 'M6']

def _mets(sub):
    ep = sub['e_rem_pct'].dropna()
    eu = sub['e_rem_usd']
    return pd.Series({
        'MAPE': ep.abs().mean() * 100,
        'RMSE_mm': np.sqrt((eu ** 2).mean()) / 1e6,
        'Bias_mm': eu.mean() / 1e6,
        'HR@10%': (ep.abs() <= 0.10).mean() * 100,
        'HR@20%': (ep.abs() <= 0.20).mean() * 100,
        'MaxAPE': ep.abs().max() * 100,
    })

def _subgrid(n=7, ncols=4):
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes

def _cone_fig(col, scale, ylim, ref, unit, ylabel, color, fname, suptitle):
    fig, axes = _subgrid()
    p_tbl = {}
    for idx, m in enumerate(MODELS):
        ax = axes[idx]
        sub = errors[errors['model'] == m]
        vals = {t: sub[sub['T'] == t][col].dropna().values * scale for t in range(18)}
        pc = {p: [np.percentile(vals[t], p) for t in range(18)] for p in [5, 25, 50, 75, 95]}
        ts = list(range(18))
        ax.fill_between(ts, pc[5], pc[95], alpha=0.2, color=color)
        ax.fill_between(ts, pc[25], pc[75], alpha=0.4, color=color)
        ax.plot(ts, pc[50], color=color, lw=2)
        ax.axhline(ref, color='red', ls='--', lw=1)
        ax.axhline(-ref, color='red', ls='--', lw=1)
        ax.set_ylim(*ylim)
        ax.set_xlim(0, 17)
        ax.set_xlabel('T')
        ax.set_ylabel(ylabel)
        first_t = next((t for t in range(18) if pc[5][t] >= -ref and pc[95][t] <= ref), None)
        lbl = (f'P90⊂±{ref}{unit}: T={first_t}' if first_t is not None
               else f'P90 never ⊂±{ref}{unit}')
        ax.set_title(f'{m}  {lbl}', fontsize=9)
        p_tbl[m] = {**{f'T{t}_P5': pc[5][t] for t in T_SUMMARY},
                    **{f'T{t}_P95': pc[95][t] for t in T_SUMMARY}}
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return pd.DataFrame(p_tbl).T

# --- Output 1: MAPE wide table ---
mape_wide = pd.DataFrame(
    {t: {m: errors[(errors['model'] == m) & (errors['T'] == t)]['e_rem_pct']
             .dropna().abs().mean() * 100
         for m in MODELS}
     for t in T_SUMMARY}
)
mape_wide.index.name = 'Model'
mape_wide.columns.name = 'T'
print('=== Output 1: MAPE (%) ===')
print(mape_wide.round(2).to_string())
print()

# --- Output 2: full metric tables ---
for t in T_SUMMARY:
    print(f'=== Output 2: Metrics T={t} ===')
    rows = [_mets(errors[(errors['model'] == m) & (errors['T'] == t)]).rename(m)
            for m in MODELS]
    print(pd.DataFrame(rows).round(3).to_string())
    print()

# --- Output 5: MAPE decay curves ---
fig, ax = plt.subplots(figsize=(10, 5))
for m in MODELS:
    y = [errors[(errors['model'] == m) & (errors['T'] == t)]['e_rem_pct']
         .dropna().abs().mean() * 100 for t in range(18)]
    ax.plot(range(18), y, marker='o', ms=3, label=m)
ax.set_xlabel('T')
ax.set_ylabel('MAPE (%)')
ax.set_title('MAPE Decay Curves')
ax.legend()
ax.set_xlim(0, 17)
fig.tight_layout()
fig.savefig('mape_decay.png', dpi=150)
plt.close(fig)

# --- Output 3: View C — percentile cone USD ---
p5p95_tbl = _cone_fig(
    'e_rem_usd', 1 / 1e6, (-150, 150), 50, 'mm', 'e_rem (USDmm)',
    'steelblue', 'view_c_cone_usd.png', 'View C — Percentile Cone (USD)',
)
print('=== Output 3: P5/P95 (USDmm) per model × T_SUMMARY ===')
print(p5p95_tbl.round(1).to_string())
print()

# --- Output 4: View D — percentile cone pct ---
_ = _cone_fig(
    'e_rem_pct', 100, (-100, 100), 20, '%', 'e_rem (%)',
    'darkorange', 'view_d_cone_pct.png', 'View D — Percentile Cone (%)',
)
mean_usd = pd.DataFrame(
    {t: {m: errors[(errors['model'] == m) & (errors['T'] == t)]['e_rem_usd'].abs().mean() / 1e6
         for m in MODELS}
     for t in T_SUMMARY}
)
mean_pct = pd.DataFrame(
    {t: {m: errors[(errors['model'] == m) & (errors['T'] == t)]['e_rem_pct'].dropna().abs().mean() * 100
         for m in MODELS}
     for t in T_SUMMARY}
)
mean_usd.index.name = mean_pct.index.name = 'Model'
mean_usd.columns.name = mean_pct.columns.name = 'T'
print('=== Output 4: Mean |e_rem| USDmm ===')
print(mean_usd.round(2).to_string())
print()
print('=== Output 4: Mean |e_rem| % ===')
print(mean_pct.round(2).to_string())
print()

# --- Output 6: View A — exceedance ---
exc = pd.DataFrame(
    {t: {m: (errors[(errors['model'] == m) & (errors['T'] == t)]['e_rem_usd'].abs() > 50e6)
             .mean() * 100
          for m in MODELS}
     for t in T_SUMMARY}
)
exc.index.name = 'Model'
exc.columns.name = 'T'
print('=== Output 6: View A — fraction |e_rem| > 50mm (%) ===')
print(exc.round(1).to_string())
print()
fails = (errors[(errors['model'] == 'M2') & (errors['T'] == 6) &
                (errors['e_rem_usd'].abs() > 50e6)]
         .sort_values('session_date'))
print(f'M2 T=6 failures ({len(fails)} sessions):')
for _, r in fails.iterrows():
    print(f'  {r["session_date"]}  {r["weekday"]}  '
          f'e={r["e_rem_usd"] / 1e6:+.1f}mm  V_true={r["V_true"] / 1e6:.1f}mm')
print()

# --- Output 7: View B — scatter ±50mm ---
fig, axes = _subgrid()
for idx, m in enumerate(MODELS):
    ax = axes[idx]
    sub = errors[(errors['model'] == m) & (errors['T'] >= 1)]
    ax.scatter(sub['T'], sub['e_rem_usd'] / 1e6, alpha=0.25, s=5,
               color='steelblue', rasterized=True)
    mean_t = sub.groupby('T')['e_rem_usd'].mean() / 1e6
    ax.plot(mean_t.index, mean_t.values, color='red', lw=1.5)
    ax.axhline(50, color='black', ls='--', lw=0.8)
    ax.axhline(-50, color='black', ls='--', lw=0.8)
    ax.set_ylim(-200, 200)
    ax.set_xlim(1, 17)
    ax.set_title(m, fontsize=9)
    ax.set_xlabel('T')
    ax.set_ylabel('e_rem (USDmm)')
fig.suptitle('View B — Scatter ±50mm', fontsize=11)
fig.tight_layout()
fig.savefig('view_b_scatter.png', dpi=150)
plt.close(fig)

# --- Output 8: pairwise t-stats M2–M6 ---
print('=== Output 8: Pairwise t-stats Δ|e_rem_pct| (positive = row worse) ===')
for t in [3, 6, 9, 12, 15]:
    sub = errors[(errors['T'] == t) & (errors['model'].isin(SM))]
    piv = sub.pivot(index='session_date', columns='model', values='e_rem_pct').abs().dropna()
    mat = pd.DataFrame(np.nan, index=SM, columns=SM)
    for m1 in SM:
        for m2 in SM:
            if m1 != m2:
                d = piv[m1].values - piv[m2].values
                mat.loc[m1, m2] = stats.ttest_1samp(d, 0).statistic
    print(f'T={t}')
    print(mat.round(3).to_string())
    print()

# --- Output 9: error trajectories T=6 ---
fig, axes = _subgrid()
for idx, m in enumerate(MODELS):
    ax = axes[idx]
    sub = errors[(errors['model'] == m) & (errors['T'] == 6)].sort_values('session_date')
    ax.plot(range(len(sub)), sub['e_rem_pct'].values * 100, color='steelblue', lw=1)
    ax.axhline(20, color='red', ls='--', lw=1)
    ax.axhline(-20, color='red', ls='--', lw=1)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_title(m, fontsize=9)
    ax.set_xlabel('Session index')
    ax.set_ylabel('e_rem (%)')
fig.suptitle('Error Trajectories — T=6', fontsize=11)
fig.tight_layout()
fig.savefig('error_traj_t6.png', dpi=150)
plt.close(fig)

# --- Output 10: abnormality score M2 ---
_t0 = (errors[(errors['model'] == 'M2') & (errors['T'] == 0)]
       [['session_date', 'weekday', 'V_hat_rem']]
       .rename(columns={'V_hat_rem': 'V_bar_wd'}))
_t6 = errors[(errors['model'] == 'M2') & (errors['T'] == 6)][
    ['session_date', 'V_realized', 'V_hat_rem']].copy()
_t6['V_hat_full'] = _t6['V_realized'] + _t6['V_hat_rem']
sc = (_t0.merge(_t6[['session_date', 'V_hat_full']], on='session_date')
        .sort_values('session_date')
        .reset_index(drop=True))
sc['score'] = sc['V_hat_full'] / sc['V_bar_wd']
s_mean, s_std = sc['score'].mean(), sc['score'].std()

wds = sc['weekday'].unique()
wd_col = dict(zip(wds, plt.cm.tab10(np.linspace(0, 1, len(wds)))))
fig, ax = plt.subplots(figsize=(14, 5))
ax.axhspan(s_mean - s_std, s_mean + s_std, alpha=0.1, color='gray')
ax.axhline(s_mean + s_std, color='gray', ls='--', lw=1)
ax.axhline(s_mean - s_std, color='gray', ls='--', lw=1)
ax.axhline(s_mean, color='gray', lw=0.8)
for wd in wds:
    sub_wd = sc[sc['weekday'] == wd]
    ax.scatter(sub_wd.index, sub_wd['score'], color=wd_col[wd], s=25, label=wd, zorder=3)
ax.set_xlabel('Session index')
ax.set_ylabel('V̂_full(T=6) / V̄_wd')
ax.set_title('Abnormality Score M2')
ax.legend(fontsize=8)
ax.set_xlim(0, len(sc) - 1)
fig.tight_layout()
fig.savefig('abnormality_m2.png', dpi=150)
plt.close(fig)

print('Figures saved: mape_decay.png  view_c_cone_usd.png  view_d_cone_pct.png  '
      'view_b_scatter.png  error_traj_t6.png  abnormality_m2.png')
