#!/usr/bin/env python3
"""
AI4I 2020 Predictive Maintenance Dataset
EDA Figure Generation Script
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

FIGURES_DIR = '/home/ubuntu/cloud-mlops/figures'
DATASET_PATH = '/home/ubuntu/cloud-mlops/dataset/ai4i2020.csv'

os.makedirs(FIGURES_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATASET_PATH)
df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]

# Feature engineering
df['Temp_Diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['Power_W'] = df['Torque [Nm]'] * (df['Rotational speed [rpm]'] * 2 * np.pi / 60)
df['Strain_min_Nm'] = df['Tool wear [min]'] * df['Torque [Nm]']

# Palette
palette_binary = {0: '#4A90E2', 1: '#E74C3C'}  # Normal: Blue, Failure: Red

print("Generating Figure 1: Target and Failure Distribution...")
# -------------------------------------------------------------
# Figure 1: Target and Failure Distribution
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Subplot 1: Machine failure distribution
mf_counts = df['Machine failure'].value_counts()
labels = ['Normal (0)', 'Failure (1)']
colors = ['#4A90E2', '#E74C3C']
bars1 = axes[0].bar(labels, [mf_counts[0], mf_counts[1]], color=colors, width=0.55, edgecolor='black', alpha=0.85)
axes[0].set_title('Machine Failure Distribution (Class Imbalance)', fontsize=13, fontweight='bold', pad=12)
axes[0].set_ylabel('Record Count', fontsize=11)
axes[0].set_ylim(0, 11000)
for bar in bars1:
    height = bar.get_height()
    pct = height / len(df) * 100
    axes[0].annotate(f'{height:,}\n({pct:.2f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

# Subplot 2: 5 Failure Modes Breakdown
modes = ['HDF', 'OSF', 'PWF', 'TWF', 'RNF']
mode_counts = [df[m].sum() for m in modes]
mode_names = ['HDF (Heat)', 'OSF (Overstrain)', 'PWF (Power)', 'TWF (Tool Wear)', 'RNF (Random)']
bars2 = axes[1].barh(mode_names[::-1], mode_counts[::-1], color='#E67E22', edgecolor='black', alpha=0.85)
axes[1].set_title('Failure Mode Breakdown (Count & % of Total)', fontsize=13, fontweight='bold', pad=12)
axes[1].set_xlabel('Failure Instances', fontsize=11)
axes[1].set_xlim(0, 140)
for bar in bars2:
    width = bar.get_width()
    pct = width / len(df) * 100
    axes[1].annotate(f' {width} ({pct:.2f}%)',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=10, fontweight='bold')

# Subplot 3: Failure Rate by Product Type
type_order = ['L', 'M', 'H']
type_totals = df['Type'].value_counts()[type_order]
type_failures = df.groupby('Type')['Machine failure'].sum()[type_order]
type_rates = (type_failures / type_totals) * 100

bars3 = axes[2].bar(['L (Low 60%)', 'M (Medium 30%)', 'H (High 10%)'], type_rates,
                    color=['#9B59B6', '#3498DB', '#2ECC71'], edgecolor='black', width=0.55, alpha=0.85)
axes[2].set_title('Failure Rate by Product Type', fontsize=13, fontweight='bold', pad=12)
axes[2].set_ylabel('Failure Rate (%)', fontsize=11)
axes[2].set_ylim(0, 5.0)
for bar, fail_cnt, tot_cnt in zip(bars3, type_failures, type_totals):
    height = bar.get_height()
    axes[2].annotate(f'{height:.2f}%\n({fail_cnt}/{tot_cnt:,})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, '01_target_and_failure_distribution.png'), dpi=200)
plt.close(fig)


print("Generating Figure 2: Sensor Features by Failure...")
# -------------------------------------------------------------
# Figure 2: Sensor Features Distribution by Failure Status
# -------------------------------------------------------------
raw_sensor_cols = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, col in enumerate(raw_sensor_cols):
    sns.boxplot(x='Machine failure', y=col, hue='Machine failure', data=df, ax=axes[i],
                palette={0: '#4A90E2', 1: '#E74C3C', '0': '#4A90E2', '1': '#E74C3C'}, width=0.45,
                legend=False,
                showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"6"})
    axes[i].set_title(col, fontsize=11, fontweight='bold', pad=10)
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(['Normal (0)', 'Failure (1)'], fontsize=10)
    axes[i].set_xlabel('')
    axes[i].set_ylabel(col, fontsize=10)

plt.suptitle('Raw Sensor Features: Normal (0) vs Machine Failure (1) (White dots = Mean)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, '02_sensor_features_by_failure.png'), dpi=200, bbox_inches='tight')
plt.close(fig)


print("Generating Figure 3: Correlation Matrix Heatmap...")
# -------------------------------------------------------------
# Figure 3: Correlation Matrix Heatmap
# -------------------------------------------------------------
corr_cols = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'Temp_Diff',
    'Power_W',
    'Strain_min_Nm',
    'Machine failure',
    'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
]

short_names = [
    'Air Temp', 'Process Temp', 'Rot Speed', 'Torque', 'Tool Wear',
    'Temp Diff (ΔT)', 'Power (W)', 'Strain (Nm·min)',
    'Machine Failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
]

corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1.0, vmax=1.0,
            xticklabels=short_names, yticklabels=short_names, ax=ax,
            cbar_kws={'label': 'Pearson Correlation Coefficient', 'shrink': 0.8},
            linewidths=0.5, annot_kws={'size': 9})
ax.set_title('Correlation Heatmap: Sensors, Engineered Features & Failure Modes', fontsize=14, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, '03_correlation_matrix.png'), dpi=200)
plt.close(fig)


print("Generating Figure 4: Physical Failure Mechanisms Scatter...")
# -------------------------------------------------------------
# Figure 4: Physical Failure Mechanisms Verification
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 13))

# Panel 1: HDF (Heat Dissipation Failure)
# ΔT < 8.6 K & Speed <= 1380 rpm
sns.scatterplot(data=df, x='Rotational speed [rpm]', y='Temp_Diff',
                hue='HDF', palette={0: '#95A5A6', 1: '#E74C3C'},
                alpha=0.6, s=25, ax=axes[0, 0])
axes[0, 0].axvline(1380, color='darkred', linestyle='--', linewidth=1.5, label='Speed <= 1380 rpm')
axes[0, 0].axhline(8.6, color='darkblue', linestyle='--', linewidth=1.5, label='ΔT < 8.6 K')
axes[0, 0].fill_between([1150, 1380], 7.0, 8.6, color='red', alpha=0.15, label='HDF Failure Zone')
axes[0, 0].set_title('A. Heat Dissipation Failure (HDF) Mechanism', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Rotational speed [rpm]', fontsize=11)
axes[0, 0].set_ylabel('Temp Diff (Process - Air) [K]', fontsize=11)
axes[0, 0].legend(loc='upper right', frameon=True, fontsize=9)
axes[0, 0].set_xlim(1150, 2900)
axes[0, 0].set_ylim(7.0, 12.5)

# Panel 2: PWF (Power Failure)
# Power < 3500 W or > 9000 W
# Power = Torque * (rpm * 2pi / 60) -> Torque = Power / (rpm * 2pi / 60)
rpm_range = np.linspace(1168, 2886, 300)
rad_s = rpm_range * 2 * np.pi / 60
torque_low = 3500 / rad_s
torque_high = 9000 / rad_s

sns.scatterplot(data=df, x='Rotational speed [rpm]', y='Torque [Nm]',
                hue='PWF', palette={0: '#95A5A6', 1: '#E74C3C'},
                alpha=0.6, s=25, ax=axes[0, 1])
axes[0, 1].plot(rpm_range, torque_low, color='blue', linestyle='--', linewidth=1.8, label='P = 3,500 W Threshold')
axes[0, 1].plot(rpm_range, torque_high, color='purple', linestyle='--', linewidth=1.8, label='P = 9,000 W Threshold')
axes[0, 1].set_title('B. Power Failure (PWF) Mechanism: P < 3500W or P > 9000W', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Rotational speed [rpm]', fontsize=11)
axes[0, 1].set_ylabel('Torque [Nm]', fontsize=11)
axes[0, 1].legend(loc='upper right', frameon=True, fontsize=9)
axes[0, 1].set_xlim(1150, 2900)
axes[0, 1].set_ylim(0, 80)

# Panel 3: OSF (Overstrain Failure)
# Tool wear * Torque > threshold (L: 11000, M: 12000, H: 13000)
wear_range = np.linspace(140, 260, 200)
t_L = 11000 / wear_range
t_M = 12000 / wear_range
t_H = 13000 / wear_range

sns.scatterplot(data=df, x='Tool wear [min]', y='Torque [Nm]',
                hue='OSF', palette={0: '#95A5A6', 1: '#E74C3C'},
                alpha=0.6, s=25, ax=axes[1, 0])
axes[1, 0].plot(wear_range, t_L, color='red', linestyle='--', linewidth=1.5, label='L Threshold (11,000)')
axes[1, 0].plot(wear_range, t_M, color='orange', linestyle='--', linewidth=1.5, label='M Threshold (12,000)')
axes[1, 0].plot(wear_range, t_H, color='green', linestyle='--', linewidth=1.5, label='H Threshold (13,000)')
axes[1, 0].set_title('C. Overstrain Failure (OSF) Mechanism: Wear × Torque > Threshold', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Tool wear [min]', fontsize=11)
axes[1, 0].set_ylabel('Torque [Nm]', fontsize=11)
axes[1, 0].legend(loc='upper left', frameon=True, fontsize=9)
axes[1, 0].set_xlim(0, 260)
axes[1, 0].set_ylim(0, 80)

# Panel 4: TWF (Tool Wear Failure)
axes[1, 1].hist(df[df['TWF'] == 0]['Tool wear [min]'], bins=30, alpha=0.5, color='#4A90E2', label='Normal (TWF=0)', density=True)
axes[1, 1].hist(df[df['TWF'] == 1]['Tool wear [min]'], bins=15, alpha=0.8, color='#E74C3C', label='Tool Wear Failure (TWF=1)', density=True)
axes[1, 1].axvspan(198, 253, color='red', alpha=0.15, label='TWF Critical Zone (200-240 min)')
axes[1, 1].set_title('D. Tool Wear Failure (TWF) Density Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Tool wear [min]', fontsize=11)
axes[1, 1].set_ylabel('Density', fontsize=11)
axes[1, 1].legend(loc='upper left', frameon=True, fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, '04_physical_failure_mechanisms.png'), dpi=200)
plt.close(fig)


print("Generating Figure 5: Engineered Features Distribution...")
# -------------------------------------------------------------
# Figure 5: Engineered Features Distribution (Normal vs Failure)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

eng_features = [
    ('Temp_Diff', 'Temperature Difference ΔT [K]', 'Process Temp - Air Temp'),
    ('Power_W', 'Mechanical Power [W]', 'Torque × Rotational Speed'),
    ('Strain_min_Nm', 'Overstrain Index [min·Nm]', 'Tool Wear × Torque')
]

for i, (col, title, formula) in enumerate(eng_features):
    sns.kdeplot(data=df[df['Machine failure'] == 0][col], ax=axes[i],
               color='#4A90E2', label='Normal (0)', fill=True, alpha=0.3, linewidth=2)
    sns.kdeplot(data=df[df['Machine failure'] == 1][col], ax=axes[i],
               color='#E74C3C', label='Machine Failure (1)', fill=True, alpha=0.4, linewidth=2)
    axes[i].set_title(f'{title}\n({formula})', fontsize=12, fontweight='bold', pad=10)
    axes[i].set_xlabel(col, fontsize=11)
    axes[i].set_ylabel('Density', fontsize=11)
    axes[i].legend(loc='upper right', frameon=True, fontsize=10)

plt.suptitle('Engineered Physical Features: Normal vs Machine Failure Distribution', fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, '05_engineered_features_distribution.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

print("All figures successfully generated in", FIGURES_DIR)
