"""
HGD Dataset — Exploratory Data Analysis
Saves all plots as PNG images alongside this script.
"""

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from braindecode.datasets import HGD

mne.set_log_level('WARNING')
mne.set_config('MNE_DATA', '/home/glider/mne_data')

# ── Config ────────────────────────────────────────────────────────────────────
SUBJECT_IDS  = None   # all 14 subjects
TMIN, TMAX   = -0.5, 4.0
BASELINE     = (-0.5, 0)
MI_CHANNELS  = ['C3', 'Cz', 'C4']
ALL_CLASSES  = ['left_hand', 'right_hand', 'feet', 'rest']
SFREQ        = 500

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading HGD dataset...")
dataset = HGD(subject_ids=SUBJECT_IDS)
print(f"Loaded {len(dataset.datasets)} recordings.\n")


# ── EDA 1: Dataset Structure Overview ────────────────────────────────────────
print("=== Dataset Structure ===")
rows = []
for ds in dataset.datasets:
    raw  = ds.raw
    desc = ds.description
    rows.append({
        'Subject'           : int(desc['subject']),
        'Split'             : desc['run'].replace('0', '').replace('1', ''),
        'Duration (s)'      : round(raw.times[-1], 2),
        'Duration (min)'    : round(raw.times[-1] / 60, 2),
        'Sampling rate (Hz)': int(raw.info['sfreq']),
        'N channels'        : len(raw.ch_names),
        'N annotations'     : len(raw.annotations),
    })

df_structure = (
    pd.DataFrame(rows)
    .sort_values(['Subject', 'Split'])
    .reset_index(drop=True)
)
print(df_structure.to_string(index=False))


# ── EDA 2: Trial Counts per Class ─────────────────────────────────────────────
print("\n=== Trial Counts per Class ===")
rows = []
for ds in dataset.datasets:
    raw    = ds.raw
    desc   = ds.description
    counts = Counter(raw.annotations.description)
    row    = {
        'Subject': int(desc['subject']),
        'Split'  : desc['run'].replace('0', '').replace('1', ''),
    }
    for cls in ALL_CLASSES:
        row[cls] = counts.get(cls, 0)
    row['Total']    = sum(counts.values())
    row['MI total'] = counts.get('left_hand', 0) + counts.get('right_hand', 0)
    rows.append(row)

df_counts = (
    pd.DataFrame(rows)
    .sort_values(['Subject', 'Split'])
    .reset_index(drop=True)
)
print(df_counts.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4))
x       = np.arange(len(df_counts))
width   = 0.35
bars_l  = ax.bar(x - width / 2, df_counts['left_hand'],  width, label='Left hand',  color='#1f77b4')
bars_r  = ax.bar(x + width / 2, df_counts['right_hand'], width, label='Right hand', color='#d62728')
ax.set_xticks(x)
ax.set_xticklabels(
    [f"S{r['Subject']}\n{r['Split']}" for _, r in df_counts.iterrows()],
    fontsize=9,
)
ax.set_ylabel('Number of trials')
ax.set_title('MI Trial Counts — Left vs Right Hand per Subject & Split')
ax.legend()
ax.grid(axis='y', alpha=0.3)
for bar in list(bars_l) + list(bars_r):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        str(int(bar.get_height())),
        ha='center', va='bottom', fontsize=8,
    )
plt.tight_layout()
plt.savefig('mi_trial_counts_2.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: mi_trial_counts_2.png")


# ── EDA 3: Epoch Summary & Signal Stats ───────────────────────────────────────
print("\n=== Epoch Summary ===")
rows = []
for ds in dataset.datasets:
    raw    = ds.raw
    desc   = ds.description
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    mi_id  = {k: v for k, v in event_id.items() if k in ('left_hand', 'right_hand')}
    epochs = mne.Epochs(
        raw, events, event_id=mi_id,
        tmin=0.0, tmax=4.0,
        baseline=None, preload=True, verbose=False,
    )
    data = epochs.get_data()
    rows.append({
        'Subject'             : int(desc['subject']),
        'Split'               : desc['run'].replace('0', '').replace('1', ''),
        'MI epochs'           : data.shape[0],
        'Samples/epoch'       : data.shape[2],
        'Channels'            : data.shape[1],
        'Epoch duration (s)'  : round(4.0 - 0.0, 1),
        'Mean amplitude (µV)' : round(data.mean() * 1e6, 4),
        'Std amplitude (µV)'  : round(data.std()  * 1e6, 4),
        'Min (µV)'            : round(data.min()  * 1e6, 2),
        'Max (µV)'            : round(data.max()  * 1e6, 2),
    })

df_epochs = (
    pd.DataFrame(rows)
    .sort_values(['Subject', 'Split'])
    .reset_index(drop=True)
)
print(df_epochs.to_string(index=False))
total_mi = df_epochs['MI epochs'].sum()
print(f"\nTotal MI epochs : {total_mi}  (~{total_mi // 2} left / {total_mi // 2} right)")


# ── EDA 4: Amplitude Distribution (Subject 1, C3 & C4) ───────────────────────
raw0     = dataset.datasets[0].raw
ev, eid  = mne.events_from_annotations(raw0, verbose=False)
mi_id    = {k: v for k, v in eid.items() if k in ('left_hand', 'right_hand')}
ep       = mne.Epochs(
    raw0, ev, event_id=mi_id,
    tmin=0, tmax=4,
    baseline=None, picks=['C3', 'C4'], preload=True, verbose=False,
)
data_uv = ep.get_data() * 1e6

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Amplitude Distribution — MI Epochs, Subject 1', fontsize=12)
for ax, ch_idx, ch_name in zip(axes, [0, 1], ['C3', 'C4']):
    left_vals  = data_uv[ep.events[:, 2] == eid['left_hand'],  ch_idx, :].ravel()
    right_vals = data_uv[ep.events[:, 2] == eid['right_hand'], ch_idx, :].ravel()
    ax.hist(left_vals,  bins=100, alpha=0.6, color='#1f77b4', label='Left hand',  density=True)
    ax.hist(right_vals, bins=100, alpha=0.6, color='#d62728', label='Right hand', density=True)
    ax.set_title(f'Channel {ch_name}')
    ax.set_xlabel('Amplitude (µV)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('mi_amplitude_dist_2.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: mi_amplitude_dist_2.png")


# ── EDA 5: Trial-averaged EEG per channel (Subject 1) ────────────────────────
raw      = dataset.datasets[0].raw
events, event_id = mne.events_from_annotations(raw)
mi_event_id = {k: v for k, v in event_id.items() if k in ('left_hand', 'right_hand')}
epochs   = mne.Epochs(
    raw, events,
    event_id=mi_event_id,
    tmin=TMIN, tmax=TMAX,
    baseline=BASELINE,
    picks=MI_CHANNELS,
    preload=True, verbose=False,
)
left   = epochs['left_hand'].get_data()
right  = epochs['right_hand'].get_data()
times  = epochs.times

colors = {'left_hand': '#1f77b4', 'right_hand': '#d62728'}

fig, axes = plt.subplots(len(MI_CHANNELS), 1, figsize=(14, 8), sharex=True)
fig.suptitle('Average MI EEG Signal — C3 / Cz / C4  (Subject 1)', fontsize=13)
for i, (ax, ch) in enumerate(zip(axes, MI_CHANNELS)):
    ax.plot(times, left[:, i, :].mean(axis=0)  * 1e6, color=colors['left_hand'],
            label='Left hand',  linewidth=1.5)
    ax.plot(times, right[:, i, :].mean(axis=0) * 1e6, color=colors['right_hand'],
            label='Right hand', linewidth=1.5)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.8, label='Cue onset')
    ax.axvspan(0, 4, alpha=0.06, color='gray', label='MI period')
    ax.set_ylabel(f'{ch}\n(µV)', fontsize=9)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(loc='upper right', fontsize=8)
axes[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig('mi_avg_signal_2.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: mi_avg_signal_2.png")


# ── EDA 6: Single-trial heatmap at C3 (Subject 1) ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
fig.suptitle('Single-trial EEG at C3 — Left vs Right Hand  (Subject 1)', fontsize=13)
vmax = np.percentile(np.abs(np.concatenate([left[:, 0, :], right[:, 0, :]])), 95) * 1e6
for ax, data, label, color in zip(
    axes,
    [left[:, 0, :], right[:, 0, :]],
    ['Left hand', 'Right hand'],
    [colors['left_hand'], colors['right_hand']],
):
    im = ax.imshow(
        data * 1e6,
        aspect='auto',
        extent=[times[0], times[-1], 0, data.shape[0]],
        origin='lower',
        cmap='RdBu_r',
        vmin=-vmax, vmax=vmax,
    )
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    ax.set_title(label, color=color, fontsize=11)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trial #')
    plt.colorbar(im, ax=ax, label='µV')
plt.tight_layout()
plt.savefig('mi_single_trials_2.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: mi_single_trials_2.png")

print("\nAll EDA images saved.")
