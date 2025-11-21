import pandas as pd
import numpy as np
from pathlib import Path

base = Path('./conv_ppo_bc_300k')
files = {
    'trained_det': base / 'trained_det_thresh_0p8_1000.csv',
    'trained_stoch': base / 'trained_stoch_thresh_0p8_1000.csv',
}
rows = []
for policy, fp in files.items():
    df = pd.read_csv(fp)
    # normalize column names to the names used by different runners
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={
        'true_label': 'true',
        'predicted_label': 'pred',
        'confidence': 'conf',
        'pixels_seen': 'pixels',
        'episode': 'episode',
    })
    max_moves = 500
    df['triggered'] = df['moves'] < max_moves
    total = len(df)
    trig = int(df['triggered'].sum())
    prop_trig = trig/total
    def stats(subdf):
        if len(subdf)==0:
            return dict(prop_correct=float('nan'), mean_moves=float('nan'), mean_pixels=float('nan'), mean_conf=float('nan'))
        prop_correct = (subdf['pred']==subdf['true']).mean()
        return dict(prop_correct=float(prop_correct), mean_moves=float(subdf['moves'].mean()), mean_pixels=float(subdf['pixels'].mean()), mean_conf=float(subdf['conf'].mean()))
    s_tr = stats(df[df['triggered']])
    s_not = stats(df[~df['triggered']])
    rows.append({
        'policy': policy,
        'total_episodes': total,
        'triggered_count': trig,
        'triggered_prop': prop_trig,
        'triggered_prop_correct': s_tr['prop_correct'],
        'triggered_mean_moves': s_tr['mean_moves'],
        'triggered_mean_pixels': s_tr['mean_pixels'],
        'triggered_mean_conf': s_tr['mean_conf'],
        'not_triggered_count': total-trig,
        'not_triggered_prop_correct': s_not['prop_correct'],
        'not_triggered_mean_moves': s_not['mean_moves'],
        'not_triggered_mean_pixels': s_not['mean_pixels'],
        'not_triggered_mean_conf': s_not['mean_conf'],
    })
summary = pd.DataFrame(rows)
summary_fp = base / 'policy_summary_thresh_0p8.csv'
summary.to_csv(summary_fp, index=False)
print('Wrote summary to', summary_fp)
print(summary.to_string(index=False))
