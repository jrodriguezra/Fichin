import pandas as pd
import numpy as np
import glob

_found = (
    glob.glob('fichinH125.xlsx')
    + glob.glob('/home/**/fichinH125.xlsx', recursive=True)
    + glob.glob('/tmp/**/fichinH125.xlsx', recursive=True)
    + glob.glob('/data/**/fichinH125.xlsx', recursive=True)
)
XLSX = _found[0] if _found else 'fichinH125.xlsx'

df = pd.read_excel(XLSX)
df.columns = ['Fecha', 'Dia', 'Tiempo', 'Precio', 'Monto', 'Reg']

r_mask = df['Reg'] == 'R'
print(f'Reg==R rows: {r_mask.sum()}, Monto: {df.loc[r_mask, "Monto"].sum():.2f}')
df = df[~r_mask].copy()

df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True).dt.date

def _to_min(t):
    if pd.isna(t):
        return np.nan
    if isinstance(t, str):
        p = t.strip().split(':')
        return int(p[0]) * 60 + int(p[1]) + (int(p[2]) if len(p) > 2 else 0) / 60
    if hasattr(t, 'hour'):
        return t.hour * 60 + t.minute + getattr(t, 'second', 0) / 60
    try:
        return pd.Timedelta(t).total_seconds() / 60
    except Exception:
        return np.nan

df['minutes'] = df['Tiempo'].apply(_to_min)
df = df[(df['minutes'] >= 540) & (df['minutes'] < 810)].copy()
df['bin_idx'] = ((df['minutes'] - 540) // 15).astype(int).clip(0, 17)

BIN_LABELS = [f'{(540 + i * 15) // 60:02d}:{(540 + i * 15) % 60:02d}' for i in range(18)]
df['bin_t'] = df['bin_idx'].map(dict(enumerate(BIN_LABELS)))

v_td = (
    df.groupby(['Fecha', 'bin_t'])['Monto'].sum()
    .unstack(fill_value=0)
    .reindex(columns=BIN_LABELS, fill_value=0)
)
V_d = v_td.sum(axis=1)
s_td = v_td.div(V_d, axis=0)

df['PxM'] = df['Precio'] * df['Monto']
_gb = df.groupby(['Fecha', 'bin_t'])
P_vwap_td = (
    (_gb['PxM'].sum() / _gb['Monto'].sum())
    .unstack()
    .reindex(columns=BIN_LABELS)
    .replace(0, np.nan)
    .ffill(axis=1)
    .bfill(axis=1)
)
_gf = df.groupby('Fecha')
VWAP_d = _gf['PxM'].sum() / _gf['Monto'].sum()

print(f'Rows: {len(df)}, Sessions: {len(V_d)}, Zero bins: {(v_td == 0).values.sum()}')
print(f'V(d) mean={V_d.mean() / 1e6:.3f}M  min={V_d.min() / 1e6:.3f}M  max={V_d.max() / 1e6:.3f}M')

bp = P_vwap_td.stack(dropna=False).rename('P_vwap').reset_index()
bp.columns = ['session_date', 'bin_t', 'P_vwap']
bp['VWAP_session'] = bp['session_date'].map(VWAP_d)
bp.to_pickle('bin_prices.pkl')

# -- Walk-forward --

dia_map = df.groupby('Fecha')['Dia'].first()
sessions = V_d.index.tolist()
OOS_START = 61
SM = ['M2', 'M3', 'M4', 'M5', 'M6']

def _mask(m, wd, wds):
    if m == 'M2':
        return pd.Series(True, index=wds.index)
    if m == 'M3':
        return wds == wd
    if m == 'M4':
        return (wds == 'Viernes') if wd == 'Viernes' else (wds != 'Viernes')
    if m == 'M5':
        g = ['Lunes', 'Viernes']
        return wds.isin(g) if wd in g else ~wds.isin(g)
    if m == 'M6':
        return (wds == 'Jueves') if wd == 'Jueves' else (wds != 'Jueves')

main_rows, share_rows = [], []

for oi in range(OOS_START, len(sessions)):
    date = sessions[oi]
    wd = dia_map[date]
    tr = sessions[:oi]
    V_tr = V_d[tr]
    wds_tr = dia_map[tr]
    V_bar = float(V_tr.mean())
    V_bar_wd = float(V_tr[wds_tr == wd].mean())
    s_tr = s_td.loc[tr]
    mu = {m: s_tr.loc[_mask(m, wd, wds_tr)].mean() for m in SM}

    for T in range(18):
        V_real = float(v_td.loc[date, BIN_LABELS[:T]].sum()) if T > 0 else 0.0
        V_rem = float(v_td.loc[date, BIN_LABELS[T:]].sum())
        V_true = float(V_d[date])

        for m in ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6']:
            if m == 'M0':
                Vh = V_bar if T == 0 else max(V_bar - V_real, 0.0)
            elif m == 'M1':
                Vh = V_bar_wd if T == 0 else max(V_bar_wd - V_real, 0.0)
            else:
                mm = mu[m]
                mu_rem = float(mm.iloc[T:].sum())
                if T == 0:
                    Vh = V_bar_wd
                else:
                    mu_el = float(mm.iloc[:T].sum())
                    Vh = V_real * mu_rem / mu_el if mu_el > 0 else V_bar_wd * mu_rem

            e_usd = Vh - V_rem
            e_pct = e_usd / V_rem if V_rem > 0 else np.nan
            main_rows.append({
                'session_date': date, 'weekday': wd, 'T': T, 'model': m,
                'V_true': V_true, 'V_realized': V_real, 'V_rem_true': V_rem,
                'V_hat_rem': Vh, 'e_rem_usd': e_usd, 'e_rem_pct': e_pct,
            })

            if m in SM:
                mm = mu[m]
                mu_rem = float(mm.iloc[T:].sum())
                rem_bl = BIN_LABELS[T:]
                v_rem = v_td.loc[date, rem_bl]
                v_rem_tot = float(v_rem.sum())
                for i, bt in enumerate(rem_bl):
                    sh = float(mm.iloc[T + i]) / mu_rem if mu_rem > 0 else 0.0
                    st = float(v_rem[bt]) / v_rem_tot if v_rem_tot > 0 else 0.0
                    share_rows.append({
                        'session_date': date, 'T': T, 'model': m, 'bin_t': bt,
                        's_hat': sh, 's_true': st, 'e_share': sh - st,
                    })

errors_main = pd.DataFrame(main_rows)
errors_share = pd.DataFrame(share_rows)
errors_main.to_pickle('errors_main.pkl')
errors_share.to_pickle('errors_share.pkl')

print(f'errors_main: {errors_main.shape}')
print(f'errors_share: {errors_share.shape}')
print(f'bin_prices: {bp.shape}')
