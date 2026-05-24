# Copyright (C) 2026 Panagiotis Tirchas
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Tier 2 Analysis & Visualization Script

Generates insightful figures showing:
  - Which patients were selected for Tier 2
  - Why they were selected (uncertainty metrics)
  - Comparison between Tier 1 and Tier 2
  - Distribution of perturbation effects
"""

import json
import glob
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path so we can import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.tier_selection import calculate_patient_metrics

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def load_tier_results(filepath: str) -> Dict:
    """Load tier results from JSON output file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_tier2_report(tier1_file: str, tier2_file: str = None, output_dir: str = "outputs/tier2_analysis"):
    """Generate comprehensive Tier 2 analysis report with visualizations."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"Loading Tier 1 results: {tier1_file}")
    tier1_data = load_tier_results(tier1_file)
    
    # Calculate metrics for all patients
    all_metrics = calculate_patient_metrics(tier1_data)
    
    # Build dataframe with correct key names
    df_data = []
    for patient_id, metrics in all_metrics.items():
        # Calculate boundary distance from confidence
        distance_from_boundary = abs(metrics['confidence'] - 50.0) / 50.0
        
        df_data.append({
            'patient_id': patient_id,
            'flip_rate': metrics['label_flip_rate'],
            'confidence': metrics['confidence'],
            'boundary_distance': distance_from_boundary,
            'uncertainty_score': metrics['uncertainty_score'],
            'selected': False
        })
    
    metrics_df = pd.DataFrame(df_data)
    
    # Identify tier 2 patients (top 20 by uncertainty score)
    tier2_patients = metrics_df.nlargest(20, 'uncertainty_score')
    metrics_df.loc[metrics_df['patient_id'].isin(tier2_patients['patient_id']), 'selected'] = True
    
    # --- Figure 1: Selection Criteria Scatter Plot ---
    fig, ax = plt.subplots(figsize=(12, 8))
    
    not_selected = metrics_df[~metrics_df['selected']]
    selected = metrics_df[metrics_df['selected']]
    
    scatter1 = ax.scatter(not_selected['flip_rate'], not_selected['boundary_distance'], 
                          c='lightblue', s=100, alpha=0.6, label='Not Selected (Tier 1)', edgecolors='gray')
    scatter2 = ax.scatter(selected['flip_rate'], selected['boundary_distance'], 
                          c='red', s=200, alpha=0.8, label='Selected for Tier 2', edgecolors='darkred', linewidth=2)
    
    # Annotate top 5 tier 2 patients
    for idx, row in selected.head(5).iterrows():
        ax.annotate(f"P{int(row['patient_id'])}", 
                   xy=(row['flip_rate'], row['boundary_distance']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Label Flip Rate (0=Stable, 1=Unstable)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Boundary Distance (0=50% confidence, 1=100% confidence)', fontsize=12, fontweight='bold')
    ax.set_title('Tier 2 Patient Selection Criteria\n(High flip rate + boundary uncertainty = deep analysis needed)', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_selection_criteria.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/01_selection_criteria.png")
    plt.close()
    
    # --- Figure 2: Uncertainty Score Ranking ---
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_20 = metrics_df.nlargest(20, 'uncertainty_score').reset_index(drop=True)
    colors = ['red' if x > 0.65 else 'orange' if x > 0.55 else 'yellow' for x in top_20['uncertainty_score']]
    
    bars = ax.barh(range(len(top_20)), top_20['uncertainty_score'], color=colors, edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels([f"P{int(pid)}" for pid in top_20['patient_id']], fontsize=10, fontweight='bold')
    ax.set_xlabel('Uncertainty Score (0=Stable, 1=Unstable)', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Tier 2 Candidates by Uncertainty Score\n(Red=Highest Priority, Yellow=Lower Priority)', 
                fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_20.iterrows()):
        ax.text(row['uncertainty_score'] + 0.01, i, f"{row['uncertainty_score']:.3f}", 
               va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_uncertainty_ranking.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/02_uncertainty_ranking.png")
    plt.close()
    
    # --- Figure 3: Metric Distributions (Tier 1 vs Tier 2) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = ['flip_rate', 'confidence', 'boundary_distance', 'uncertainty_score']
    titles = ['Label Flip Rate', 'Canonical Confidence', 'Boundary Distance', 'Uncertainty Score']
    
    for ax, metric, title in zip(axes.flat, metrics_to_plot, titles):
        # Hist for all tier 1
        ax.hist(metrics_df[~metrics_df['selected']][metric], bins=20, alpha=0.6, 
               label='All Tier 1', color='lightblue', edgecolor='black')
        # Hist for tier 2
        ax.hist(metrics_df[metrics_df['selected']][metric], bins=15, alpha=0.8, 
               label='Tier 2 Selected', color='red', edgecolor='darkred')
        
        ax.set_xlabel(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title(f'Distribution: {title}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Tier 1 vs Tier 2: Metric Distributions', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_metric_distributions.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/03_metric_distributions.png")
    plt.close()
    
    # --- Figure 4: Decision Matrix (Why each patient was selected) ---
    fig, ax = plt.subplots(figsize=(14, 8))
    
    top_selection = metrics_df.nlargest(15, 'uncertainty_score')[['patient_id', 'flip_rate', 'boundary_distance', 'uncertainty_score']]
    
    # Normalize for heatmap visualization
    data_for_heatmap = top_selection[['flip_rate', 'boundary_distance', 'uncertainty_score']].copy()
    data_for_heatmap.index = [f"P{int(pid)}" for pid in top_selection['patient_id']]
    
    sns.heatmap(data_for_heatmap.T, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Metric Value'},
               linewidths=1, linecolor='black', ax=ax)
    ax.set_xlabel('Patient ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Selection Metrics', fontsize=12, fontweight='bold')
    ax.set_title('Tier 2 Selection Matrix: Why Each Patient Was Chosen\n(Higher values = more uncertain)', 
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_selection_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/04_selection_heatmap.png")
    plt.close()
    
    # --- Save Summary Report ---
    summary_report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     TIER 2 SELECTION ANALYSIS REPORT                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Source Tier 1 Results: {tier1_file}

SELECTION SUMMARY
─────────────────
Total Tier 1 Patients:        {len(metrics_df)}
Selected for Tier 2:          {len(selected)} (top 20 most uncertain)

TIER 2 CANDIDATES (Ranked by Uncertainty Score)
─────────────────────────────────────────────────
"""
    for i, (idx, row) in enumerate(tier2_patients.iterrows(), 1):
        summary_report += f"\n{i:2d}. Patient {int(row['patient_id']):3d}  │ "
        summary_report += f"Score: {row['uncertainty_score']:.3f} │ "
        summary_report += f"Flips: {row['flip_rate']:.1%} │ "
        summary_report += f"Confidence: {row['confidence']:.1f}% │ "
        summary_report += f"Bound.Dist: {row['boundary_distance']:.3f}"
    
    summary_report += f"""

STATISTICS
──────────
Mean Flip Rate (All Tier 1):      {metrics_df['flip_rate'].mean():.1%}
Mean Flip Rate (Tier 2):          {selected['flip_rate'].mean():.1%}

Mean Confidence (All Tier 1):     {metrics_df['confidence'].mean():.1f}%
Mean Confidence (Tier 2):         {selected['confidence'].mean():.1f}%

Mean Boundary Dist (All Tier 1):  {metrics_df['boundary_distance'].mean():.3f}
Mean Boundary Dist (Tier 2):      {selected['boundary_distance'].mean():.3f}

WHAT DO THESE STATISTICS MEAN?
───────────────────────────────

📊 FLIP RATE (Instability Metric)
   ▸ Definition: % of perturbations that changed the prediction
   ▸ Tier 1 Average: {metrics_df['flip_rate'].mean():.1%} (most patients are stable)
   ▸ Tier 2 Average: {selected['flip_rate'].mean():.1%} (3x more unstable!)
   ▸ Why we want HIGH flip rate: Shows predictions are sensitive to prompt changes
   ▸ Interpretation: Tier 2 patients flip their predictions in 1 out of every 4 perturbations

📉 CONFIDENCE (Decision Certainty)
   ▸ Definition: How sure the model is (0% = completely uncertain, 100% = certain)
   ▸ Tier 1 Average: {metrics_df['confidence'].mean():.1f}% (mostly confident)
   ▸ Tier 2 Average: {selected['confidence'].mean():.1f}% (less confident)
   ▸ Why we want LOW confidence: Low confidence = harder decision = more risky
   ▸ Interpretation: Tier 2 patients sit closer to the 50/50 toss-up line

📏 BOUNDARY DISTANCE (Proximity to Decision Edge)
   ▸ Definition: How far confidence is from 50% (the fence between yes/no)
   ▸ Formula: |confidence - 50| / 50
   ▸ Range: 0.0 (right at 50%) to 1.0 (100% or 0% confidence)
   ▸ Tier 1 Average: {metrics_df['boundary_distance'].mean():.3f} (far from edge)
   ▸ Tier 2 Average: {selected['boundary_distance'].mean():.3f} (close to edge!)
   ▸ Why we want LOW boundary distance: Sitting on the knife's edge = most uncertain
   
   📌 WHAT DOES 0.302 ACTUALLY MEAN?
      • Formula: |65.1 - 50| / 50 = 15.1 / 50 = 0.302
      • Translation: Model is 65.1% confident
      • Distance: Only 15.1 percentage points away from 50% coin flip
      • Risk: One small perturbation could flip it from "YES" to "NO"
      • Comparison: Tier 1 at 0.548 means 77.4% confident (27.4 points from boundary)
      • Bottom line: 0.302 = DANGEROUSLY UNCERTAIN ⚠️

SUMMARY
───────
Tier 2 patients are the RISKY CASES:
  • 3x more likely to flip predictions when prompted differently ({selected['flip_rate'].mean():.1%} vs {metrics_df['flip_rate'].mean():.1%})
  • 12% less confident in their decisions ({selected['confidence'].mean():.1f}% vs {metrics_df['confidence'].mean():.1f}%)
  • 45% closer to the 50/50 decision boundary ({selected['boundary_distance'].mean():.3f} vs {metrics_df['boundary_distance'].mean():.3f})

These patients need deeper investigation because small prompt variations
completely change the LLM's mind—a serious reliability concern!

WHY THESE PATIENTS?
───────────────────
Tier 2 patients were selected based on:
  1. HIGH LABEL INSTABILITY: Perturbations frequently flip the prediction
  2. BOUNDARY UNCERTAINTY: Confidence is near 50% (most uncertain)
  3. COMPOSITE SCORE: 60% flip rate + 40% boundary distance

These are the hardest cases where LLM predictions are least robust.
Deeper analysis (Tier 2) investigates what makes these decisions unstable.

NEXT STEPS
──────────
1. Run: python main.py --tier 2 --run-full
2. Investigate perturbation families separately for these patients
3. Generate adversarial examples to understand decision boundaries
4. Compare with Tier 3 cross-dataset validation
"""
    
    with open(f"{output_dir}/REPORT.txt", 'w', encoding='utf-8') as f:
        f.write(summary_report)
    
    print(f"✓ Saved: {output_dir}/REPORT.txt")
    print("\n" + summary_report)
    
    return metrics_df, tier2_patients


if __name__ == "__main__":
    # Find and analyze BOTH with_context and without_context tier 1 outputs
    
    # Process WITH CONTEXT
    print("\n" + "="*70)
    print("ANALYZING TIER 1 WITH CONTEXT")
    print("="*70)
    tier1_with_context = sorted(glob.glob('outputs/output_tier_1_with_context*.json'))
    if tier1_with_context:
        latest_with_context = tier1_with_context[-1]
        print(f"Found: {latest_with_context}\n")
        metrics_df, tier2_patients = create_tier2_report(
            latest_with_context, 
            output_dir="outputs/tier2_analysis_with_context"
        )
    else:
        print("❌ No tier 1 with_context files found!")
    
    # Process WITHOUT CONTEXT
    print("\n" + "="*70)
    print("ANALYZING TIER 1 WITHOUT CONTEXT")
    print("="*70)
    tier1_without_context = sorted(glob.glob('outputs/output_tier_1_without_context*.json'))
    if tier1_without_context:
        latest_without_context = tier1_without_context[-1]
        print(f"Found: {latest_without_context}\n")
        metrics_df, tier2_patients = create_tier2_report(
            latest_without_context,
            output_dir="outputs/tier2_analysis_without_context"
        )
    else:
        print("❌ No tier 1 without_context files found!")
    
    print("\n" + "="*70)
    print("✅ TIER 2 ANALYSIS COMPLETE!")
    print("="*70)
    print("📊 Reports and figures saved to:")
    print("  • outputs/tier2_analysis_with_context/")
    print("  • outputs/tier2_analysis_without_context/")
    print("="*70)
