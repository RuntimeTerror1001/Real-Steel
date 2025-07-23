#!/usr/bin/env python3
"""
Generate Final Summary Chart for Real-Steel Analysis
"""

import matplotlib.pyplot as plt
import numpy as np

def create_final_summary():
    """Create a final summary chart"""
    plt.style.use('dark_background')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Real-Steel IK Analysis - Final Summary Dashboard', fontsize=18, fontweight='bold')
    
    # Colors
    analytical_color = '#FF6B6B'
    brpso_color = '#4ECDC4'
    
    # Chart 1: Performance Metrics Comparison
    metrics = ['Speed\n(ms)', 'Accuracy\n(mm)', 'Success\n(%)', 'Reliability']
    analytical_values = [4.0, 5.0, 35.0, 6.0]
    brpso_values = [443.0, 0.1, 95.0, 9.5]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, analytical_values, width, label='Analytical IK', 
                   color=analytical_color, alpha=0.8)
    bars2 = ax1.bar(x + width/2, brpso_values, width, label='BRPSO IK', 
                   color=brpso_color, alpha=0.8)
    
    ax1.set_title('Performance Metrics Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Success Rate by Category
    categories = ['Simple', 'Boxing', 'Near Limits', 'Complex']
    analytical_success = [67, 0, 0, 20]
    brpso_success = [100, 80, 90, 85]
    
    x2 = np.arange(len(categories))
    bars3 = ax2.bar(x2 - width/2, analytical_success, width, label='Analytical IK', 
                   color=analytical_color, alpha=0.8)
    bars4 = ax2.bar(x2 + width/2, brpso_success, width, label='BRPSO IK', 
                   color=brpso_color, alpha=0.8)
    
    ax2.set_title('Success Rate by Scenario Category (%)', fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Chart 3: Key Performance Indicators
    kpis = ['Accuracy\nImprovement', 'Speed\nTrade-off', 'Success Rate\nImprovement']
    kpi_values = [98.6, -111, 171]  # %, times, %
    colors = [brpso_color if v > 0 else analytical_color for v in kpi_values]
    
    bars5 = ax3.bar(kpis, np.abs(kpi_values), color=colors, alpha=0.8)
    ax3.set_title('Key Performance Indicators', fontweight='bold')
    ax3.set_ylabel('Percentage (%)')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars5, kpi_values):
        height = bar.get_height()
        sign = '+' if val > 0 else '-'
        ax3.annotate(f'{sign}{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
    
    # Chart 4: Summary Table
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = [
        ['Metric', 'Analytical IK', 'BRPSO IK', 'Winner'],
        ['Speed (ms)', '4.0', '443.0', 'Analytical'],
        ['Accuracy (mm)', '5.0', '0.1', 'BRPSO'],
        ['Success Rate (%)', '35', '95', 'BRPSO'],
        ['Boxing Scenarios', '0%', '80%', 'BRPSO'],
        ['Near-Limit Poses', '0%', '90%', 'BRPSO'],
        ['Overall Winner', '❌', '✅', 'BRPSO']
    ]
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style the table
    for i in range(4):
        table[(0, i)].set_facecolor(brpso_color)
        table[(0, i)].set_text_props(weight='bold')
    
    # Highlight winners
    for i in range(1, 7):
        if table_data[i][3] == 'BRPSO':
            table[(i, 3)].set_facecolor(brpso_color)
        elif table_data[i][3] == 'Analytical':
            table[(i, 3)].set_facecolor(analytical_color)
    
    ax4.set_title('Performance Summary Table', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('analysis/08_summary_dashboard.png', dpi=300, bbox_inches='tight', 
               facecolor='#2C3E50')
    plt.close()
    print("  ✓ Summary Dashboard saved")

if __name__ == "__main__":
    create_final_summary()
    print("✅ Final summary chart generated!") 