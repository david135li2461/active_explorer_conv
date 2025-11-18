import pandas as pd
import numpy as np

paths = {
    'trained_det': 'active_explorer_conv/conv_ppo_resume_300k/trained_det_thresh_0p8_1000.csv',
    'trained_stoch': 'active_explorer_conv/conv_ppo_resume_300k/trained_stoch_thresh_0p8_1000.csv',
    'random': 'active_explorer_conv/conv_ppo_resume_300k/random_policy_thresh_0p8_1000.csv',
    'flood': 'active_explorer_conv/conv_ppo_resume_300k/flood_policy_thresh_0p8_1000.csv'
}

results = {}
for name,p in paths.items():
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    # infer max_moves as the maximum moves value observed (timeouts will equal this)
    max_moves = int(df['moves'].max())
    triggered = df['moves'] < max_moves
    n = len(df)
    trig_n = int(triggered.sum())
    trig_prop = trig_n / n
    stats = {}
    for grp_name, mask in [('triggered', triggered), ('not_triggered', ~triggered)]:
        sub = df[mask]
        if len(sub)==0:
            stats[grp_name] = {'prop_correct': np.nan, 'mean_moves': np.nan, 'mean_pixels': np.nan, 'mean_conf': np.nan, 'count':0}
            continue
        prop_correct = (sub['predicted_label'] == sub['true_label']).mean()
        mean_moves = sub['moves'].mean()
        mean_pixels = sub['pixels_seen'].mean()
        mean_conf = sub['confidence'].mean()
        stats[grp_name] = {'prop_correct': float(prop_correct), 'mean_moves': float(mean_moves), 'mean_pixels': float(mean_pixels), 'mean_conf': float(mean_conf), 'count': int(len(sub))}
    results[name] = {'n': n, 'max_moves': max_moves, 'trigger_prop': float(trig_prop), 'trigger_count': trig_n, 'stats': stats}

# Print summary
for name,res in results.items():
    print(f"Policy: {name}")
    print(f"  Episodes: {res['n']}, max_moves: {res['max_moves']}")
    print(f"  Triggered: {res['trigger_count']} ({res['trigger_prop']*100:.1f}%)")
    for grp in ['triggered','not_triggered']:
        s = res['stats'][grp]
        print(f"  {grp}: count={s['count']}, prop_correct={s['prop_correct']:.3f}, mean_moves={s['mean_moves']:.1f}, mean_pixels={s['mean_pixels']:.1f}, mean_conf={s['mean_conf']:.3f}")
    print()

# Save CSV summary
rows = []
for name,res in results.items():
    for grp in ['triggered','not_triggered']:
        s = res['stats'][grp]
        rows.append({
            'policy': name,
            'group': grp,
            'count': s['count'],
            'prop_correct': s['prop_correct'],
            'mean_moves': s['mean_moves'],
            'mean_pixels': s['mean_pixels'],
            'mean_conf': s['mean_conf'],
            'trigger_prop': res['trigger_prop']
        })
pd.DataFrame(rows).to_csv('active_explorer_conv/conv_ppo_resume_300k/policy_summary_thresh_0p8.csv', index=False)
print('Wrote summary CSV: active_explorer_conv/conv_ppo_resume_300k/policy_summary_thresh_0p8.csv')
