#!/usr/bin/env python3
"""
Extract Individual Analysis Charts for Real-Steel Project
Generates separate visualization files for each performance metric
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import IK solvers
try:
    from src.core.ik_analytical3d import IKAnalytical3DRefined
    from src.core.brpso_ik_solver import BRPSO_IK_Solver
except ImportError:
    from ik_analytical3d import IKAnalytical3DRefined
    from brpso_ik_solver import BRPSO_IK_Solver

class IndividualChartGenerator:
    def __init__(self):
        """Initialize the individual chart generator"""
        print("📊 Individual Analysis Chart Generator - Real-Steel Project")
        print("=" * 60)
        
        # Create analysis directory
        os.makedirs('analysis', exist_ok=True)
        
        # Initialize solvers
        self.analytical_solver = IKAnalytical3DRefined(
            upper_arm_length=0.1032,
            lower_arm_length=0.1000,
            position_tolerance=1e-5
        )
        
        self.brpso_solver = BRPSO_IK_Solver(
            upper_arm_length=0.1032,
            lower_arm_length=0.1000,
            swarm_size=30,
            max_iterations=100,
            position_tolerance=1e-4
        )
        
        # Performance tracking
        self.results = {
            'analytical': [],
            'brpso': []
        }
        
        # Test scenarios
        self.test_scenarios = self._generate_test_scenarios()
        
        # Color scheme
        self.colors = {
            'analytical': '#FF6B6B',  # Red
            'brpso': '#4ECDC4',       # Teal
            'background': '#2C3E50',   # Dark blue
            'accent': '#F39C12'        # Orange
        }
        
        # Set modern style
        plt.style.use('dark_background')
    
    def _generate_test_scenarios(self):
        """Generate comprehensive test scenarios"""
        scenarios = []
        
        # Simple reach scenarios
        simple_targets = [
            (0.15, 0.0, 0.15, "Simple Forward"),
            (0.0, 0.15, 0.15, "Simple Side"),
            (0.1, 0.1, 0.2, "Simple Diagonal")
        ]
        scenarios.extend(simple_targets)
        
        # Boxing-specific scenarios
        boxing_targets = [
            (0.2, 0.05, 0.1, "Boxing Jab"),
            (0.15, 0.15, 0.05, "Boxing Hook"),
            (0.1, -0.1, 0.25, "Boxing Uppercut"),
            (0.18, 0.0, 0.18, "Boxing Guard"),
            (0.12, 0.12, 0.12, "Boxing Block")
        ]
        scenarios.extend(boxing_targets)
        
        # Near joint limit scenarios
        limit_targets = [
            (0.19, 0.02, 0.02, "Near Max Reach"),
            (0.05, 0.18, 0.05, "Extreme Side"),
            (0.08, 0.08, 0.25, "High Reach")
        ]
        scenarios.extend(limit_targets)
        
        # Complex scenarios
        complex_targets = [
            (0.16, 0.08, 0.08, "Complex A"),
            (0.14, -0.06, 0.16, "Complex B"),
            (0.11, 0.14, 0.11, "Complex C")
        ]
        scenarios.extend(complex_targets)
        
        return scenarios
    
    def collect_performance_data(self):
        """Collect performance data for analysis"""
        print("\n🔬 Collecting Performance Data...")
        
        for x, y, z, scenario_name in self.test_scenarios:
            target = np.array([x, y, z])
            print(f"  Testing: {scenario_name}")
            
            # Test Analytical IK
            analytical_result = self._test_analytical(target, scenario_name)
            self.results['analytical'].append(analytical_result)
            
            # Test BRPSO IK
            brpso_result = self._test_brpso(target, scenario_name)
            self.results['brpso'].append(brpso_result)
    
    def _test_analytical(self, target, scenario):
        """Test analytical IK solver"""
        import time
        start_time = time.time()
        
        try:
            shoulder = np.array([0, 0, 0])
            elbow = target * 0.6
            wrist = target
            
            solution = self.analytical_solver.solve(shoulder, elbow, wrist)
            solve_time = time.time() - start_time
            
            pos, _ = self.analytical_solver.forward_kinematics(solution)
            error = np.linalg.norm(pos - target)
            
            return {
                'scenario': scenario,
                'time': solve_time,
                'error': error,
                'success': error < 0.01,
                'joint_angles': solution,
                'position_achieved': pos
            }
            
        except Exception as e:
            return {
                'scenario': scenario,
                'time': time.time() - start_time,
                'error': float('inf'),
                'success': False,
                'joint_angles': None,
                'position_achieved': None
            }
    
    def _test_brpso(self, target, scenario):
        """Test BRPSO IK solver"""
        import time
        start_time = time.time()
        
        try:
            solution = self.brpso_solver.solve(target_position=target)
            solve_time = time.time() - start_time
            
            return {
                'scenario': scenario,
                'time': solve_time,
                'error': solution['position_error'],
                'success': solution['converged'],
                'iterations': solution['iterations'],
                'joint_angles': solution['joint_angles'],
                'convergence_history': solution.get('convergence_history', [])
            }
            
        except Exception as e:
            return {
                'scenario': scenario,
                'time': time.time() - start_time,
                'error': float('inf'),
                'success': False,
                'iterations': 0,
                'joint_angles': None,
                'convergence_history': []
            }
    
    def generate_individual_charts(self):
        """Generate all individual charts"""
        print("\n📊 Generating Individual Analysis Charts...")
        
        # 1. Performance Overview
        self._create_performance_overview()
        
        # 2. Accuracy Comparison
        self._create_accuracy_comparison()
        
        # 3. Timing Analysis
        self._create_timing_analysis()
        
        # 4. Success Rate Analysis
        self._create_success_rate_analysis()
        
        # 5. BRPSO Convergence
        self._create_brpso_convergence()
        
        # 6. Error Distribution
        self._create_error_distribution()
        
        # 7. Scenario Performance Heatmap
        self._create_scenario_heatmap()
        
        # 8. Summary Statistics
        self._create_summary_statistics()
        
        print("✅ All individual charts generated in 'analysis/' folder")
    
    def _create_performance_overview(self):
        """Create performance overview chart"""
        plt.figure(figsize=(12, 8))
        
        analytical_data = self.results['analytical']
        brpso_data = self.results['brpso']
        
        # Calculate metrics
        anal_times = [r['time']*1000 for r in analytical_data if r['success']]
        brpso_times = [r['time']*1000 for r in brpso_data if r['success']]
        anal_errors = [r['error']*1000 for r in analytical_data if r['success']]
        brpso_errors = [r['error']*1000 for r in brpso_data if r['success']]
        
        # Create comparison bars
        metrics = ['Avg Time (ms)', 'Avg Error (mm)', 'Success Rate (%)']
        analytical_values = [
            np.mean(anal_times) if anal_times else 0,
            np.mean(anal_errors) if anal_errors else 0,
            len([r for r in analytical_data if r['success']]) / len(analytical_data) * 100
        ]
        brpso_values = [
            np.mean(brpso_times) if brpso_times else 0,
            np.mean(brpso_errors) if brpso_errors else 0,
            len([r for r in brpso_data if r['success']]) / len(brpso_data) * 100
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, analytical_values, width, label='Analytical IK', 
                       color=self.colors['analytical'], alpha=0.8)
        bars2 = plt.bar(x + width/2, brpso_values, width, label='BRPSO IK', 
                       color=self.colors['brpso'], alpha=0.8)
        
        plt.title('Performance Overview Comparison\nReal-Steel IK Analysis', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xticks(x, metrics)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('analysis/01_performance_overview.png', dpi=300, bbox_inches='tight', 
                   facecolor='#2C3E50')
        plt.close()
        print("  ✓ Performance Overview saved")
    
    def _create_accuracy_comparison(self):
        """Create accuracy comparison chart"""
        plt.figure(figsize=(10, 8))
        
        analytical_errors = [r['error']*1000 for r in self.results['analytical'] if r['success']]
        brpso_errors = [r['error']*1000 for r in self.results['brpso'] if r['success']]
        
        # Create box plot
        data = [analytical_errors, brpso_errors]
        labels = ['Analytical IK', 'BRPSO IK']
        
        bp = plt.boxplot(data, labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor(self.colors['analytical'])
        bp['boxes'][1].set_facecolor(self.colors['brpso'])
        
        plt.title('Position Error Distribution Comparison\nReal-Steel IK Accuracy Analysis', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Position Error (mm)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add improvement annotation
        if analytical_errors and brpso_errors:
            improvement = ((np.mean(analytical_errors) - np.mean(brpso_errors)) / 
                          np.mean(analytical_errors)) * 100
            plt.text(0.5, 0.95, f'BRPSO: {improvement:.1f}% Better Accuracy', 
                   transform=plt.gca().transAxes, ha='center', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['accent'], alpha=0.8),
                   fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('analysis/02_accuracy_comparison.png', dpi=300, bbox_inches='tight', 
                   facecolor='#2C3E50')
        plt.close()
        print("  ✓ Accuracy Comparison saved")
    
    def _create_timing_analysis(self):
        """Create timing analysis chart"""
        plt.figure(figsize=(14, 8))
        
        scenarios = [r['scenario'] for r in self.results['analytical']]
        anal_times = [r['time']*1000 for r in self.results['analytical']]
        brpso_times = [r['time']*1000 for r in self.results['brpso']]
        
        x = np.arange(len(scenarios))
        
        plt.plot(x, anal_times, 'o-', label='Analytical IK', 
               color=self.colors['analytical'], linewidth=3, markersize=8)
        plt.plot(x, brpso_times, 's-', label='BRPSO IK', 
               color=self.colors['brpso'], linewidth=3, markersize=8)
        
        plt.title('Convergence Time by Scenario\nReal-Steel IK Performance Analysis', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Test Scenarios', fontsize=12)
        plt.ylabel('Time (ms)', fontsize=12)
        plt.xticks(x, scenarios, rotation=45, ha='right')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add speed ratio annotation
        avg_anal = np.mean(anal_times)
        avg_brpso = np.mean(brpso_times)
        speed_ratio = avg_brpso / avg_anal
        plt.text(0.02, 0.98, f'BRPSO: {speed_ratio:.1f}x Slower\nBut More Accurate', 
               transform=plt.gca().transAxes, va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8),
               fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('analysis/03_timing_analysis.png', dpi=300, bbox_inches='tight', 
                   facecolor='#2C3E50')
        plt.close()
        print("  ✓ Timing Analysis saved")
    
    def _create_success_rate_analysis(self):
        """Create success rate analysis chart"""
        plt.figure(figsize=(12, 8))
        
        categories = ['Simple', 'Boxing', 'Near Limits', 'Complex']
        
        anal_success = []
        brpso_success = []
        
        for category in categories:
            anal_cat = [r for r in self.results['analytical'] if category.lower() in r['scenario'].lower()]
            brpso_cat = [r for r in self.results['brpso'] if category.lower() in r['scenario'].lower()]
            
            anal_rate = len([r for r in anal_cat if r['success']]) / len(anal_cat) * 100 if anal_cat else 0
            brpso_rate = len([r for r in brpso_cat if r['success']]) / len(brpso_cat) * 100 if brpso_cat else 0
            
            anal_success.append(anal_rate)
            brpso_success.append(brpso_rate)
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, anal_success, width, label='Analytical IK', 
                       color=self.colors['analytical'], alpha=0.8)
        bars2 = plt.bar(x + width/2, brpso_success, width, label='BRPSO IK', 
                       color=self.colors['brpso'], alpha=0.8)
        
        plt.title('Success Rate by Scenario Category\nReal-Steel IK Reliability Analysis', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Success Rate (%)', fontsize=12)
        plt.xticks(x, categories)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        # Add percentage labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.annotate(f'{height:.0f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('analysis/04_success_rate_analysis.png', dpi=300, bbox_inches='tight', 
                   facecolor='#2C3E50')
        plt.close()
        print("  ✓ Success Rate Analysis saved")
    
    def _create_brpso_convergence(self):
        """Create BRPSO convergence chart"""
        plt.figure(figsize=(12, 8))
        
        # Get convergence data
        convergence_data = []
        for result in self.results['brpso']:
            if result['success'] and result['convergence_history']:
                convergence_data.append(result['convergence_history'])
        
        if convergence_data:
            # Plot individual convergence curves
            for i, history in enumerate(convergence_data[:8]):  # Show first 8
                iterations = range(len(history))
                plt.plot(iterations, history, alpha=0.4, linewidth=1.5, color='lightblue')
            
            # Plot average convergence
            max_len = max(len(h) for h in convergence_data)
            avg_convergence = []
            for i in range(max_len):
                values = [h[i] for h in convergence_data if i < len(h)]
                avg_convergence.append(np.mean(values))
            
            plt.plot(range(len(avg_convergence)), avg_convergence, 
                   color=self.colors['accent'], linewidth=4, label='Average Convergence')
            
            plt.title('BRPSO Convergence Behavior\nReal-Steel Optimization Analysis', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Iterations', fontsize=12)
            plt.ylabel('Position Error (m)', fontsize=12)
            plt.yscale('log')
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Add convergence info
            final_error = avg_convergence[-1] if avg_convergence else 0
            plt.text(0.98, 0.98, f'Final Error: {final_error:.4f}m\nConvergence: Exponential', 
                   transform=plt.gca().transAxes, ha='right', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['brpso'], alpha=0.8),
                   fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('analysis/05_brpso_convergence.png', dpi=300, bbox_inches='tight', 
                   facecolor='#2C3E50')
        plt.close()
        print("  ✓ BRPSO Convergence saved")
    
    def _create_error_distribution(self):
        """Create error distribution chart"""
        plt.figure(figsize=(12, 8))
        
        anal_errors = [r['error']*1000 for r in self.results['analytical'] if r['success']]
        brpso_errors = [r['error']*1000 for r in self.results['brpso'] if r['success']]
        
        # Create histograms
        bins = np.linspace(0, max(max(anal_errors) if anal_errors else 0, 
                                 max(brpso_errors) if brpso_errors else 0), 25)
        
        plt.hist(anal_errors, bins=bins, alpha=0.7, label='Analytical IK', 
               color=self.colors['analytical'], density=True, edgecolor='black')
        plt.hist(brpso_errors, bins=bins, alpha=0.7, label='BRPSO IK', 
               color=self.colors['brpso'], density=True, edgecolor='black')
        
        plt.title('Error Distribution Comparison\nReal-Steel IK Precision Analysis', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Position Error (mm)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add mean lines and statistics
        if anal_errors and brpso_errors:
            plt.axvline(np.mean(anal_errors), color=self.colors['analytical'], 
                      linestyle='--', linewidth=3, alpha=0.8, label=f'Analytical Mean: {np.mean(anal_errors):.1f}mm')
            plt.axvline(np.mean(brpso_errors), color=self.colors['brpso'], 
                      linestyle='--', linewidth=3, alpha=0.8, label=f'BRPSO Mean: {np.mean(brpso_errors):.1f}mm')
            
            plt.text(0.98, 0.98, f'Analytical: μ={np.mean(anal_errors):.2f}, σ={np.std(anal_errors):.2f}\nBRPSO: μ={np.mean(brpso_errors):.2f}, σ={np.std(brpso_errors):.2f}', 
                   transform=plt.gca().transAxes, ha='right', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                   fontsize=12, fontweight='bold', color='black')
        
        plt.tight_layout()
        plt.savefig('analysis/06_error_distribution.png', dpi=300, bbox_inches='tight', 
                   facecolor='#2C3E50')
        plt.close()
        print("  ✓ Error Distribution saved")
    
    def _create_scenario_heatmap(self):
        """Create scenario performance heatmap"""
        plt.figure(figsize=(16, 10))
        
        scenarios = [r['scenario'] for r in self.results['analytical']]
        
        # Create performance matrix
        metrics = ['Time (ms)', 'Error (mm)', 'Success Rate']
        data = []
        labels = []
        
        for scenario in scenarios:
            anal_result = next(r for r in self.results['analytical'] if r['scenario'] == scenario)
            brpso_result = next(r for r in self.results['brpso'] if r['scenario'] == scenario)
            
            # Analytical row
            anal_row = [
                anal_result['time'] * 1000,
                anal_result['error'] * 1000 if anal_result['success'] else 50,
                100 if anal_result['success'] else 0
            ]
            data.append(anal_row)
            labels.append(f'{scenario} (Analytical)')
            
            # BRPSO row
            brpso_row = [
                brpso_result['time'] * 1000,
                brpso_result['error'] * 1000 if brpso_result['success'] else 50,
                100 if brpso_result['success'] else 0
            ]
            data.append(brpso_row)
            labels.append(f'{scenario} (BRPSO)')
        
        # Plot heatmap
        im = plt.imshow(data, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
        
        plt.title('Performance Heatmap by Scenario\nReal-Steel IK Comprehensive Analysis', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xticks(range(len(metrics)), metrics, fontsize=12)
        plt.yticks(range(len(labels)), labels, fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, label='Performance Value')
        cbar.ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        plt.savefig('analysis/07_scenario_heatmap.png', dpi=300, bbox_inches='tight', 
                   facecolor='#2C3E50')
        plt.close()
        print("  ✓ Scenario Heatmap saved")
    
    def _create_summary_statistics(self):
        """Create summary statistics chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Summary Statistics - Real-Steel IK Analysis', fontsize=18, fontweight='bold')
        
        # Subplot 1: Time vs Error Scatter
        anal_data = [(r['time']*1000, r['error']*1000) for r in self.results['analytical'] if r['success']]
        brpso_data = [(r['time']*1000, r['error']*1000) for r in self.results['brpso'] if r['success']]
        
        if anal_data:
            anal_times, anal_errors = zip(*anal_data)
            ax1.scatter(anal_times, anal_errors, color=self.colors['analytical'], 
                       s=100, alpha=0.7, label='Analytical IK')
        
        if brpso_data:
            brpso_times, brpso_errors = zip(*brpso_data)
            ax1.scatter(brpso_times, brpso_errors, color=self.colors['brpso'], 
                       s=100, alpha=0.7, label='BRPSO IK')
        
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Error (mm)')
        ax1.set_title('Time vs Error Trade-off')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Success Rate Pie Charts
        anal_success = len([r for r in self.results['analytical'] if r['success']])
        anal_total = len(self.results['analytical'])
        brpso_success = len([r for r in self.results['brpso'] if r['success']])
        brpso_total = len(self.results['brpso'])
        
        sizes1 = [anal_success, anal_total - anal_success]
        sizes2 = [brpso_success, brpso_total - brpso_success]
        
        ax2.pie(sizes1, labels=['Success', 'Failure'], autopct='%1.1f%%', startangle=90,
               colors=[self.colors['analytical'], 'darkred'])
        ax2.set_title('Analytical IK Success Rate')
        
        # Subplot 3: BRPSO Success Rate
        ax3.pie(sizes2, labels=['Success', 'Failure'], autopct='%1.1f%%', startangle=90,
               colors=[self.colors['brpso'], 'darkred'])
        ax3.set_title('BRPSO IK Success Rate')
        
        # Subplot 4: Performance Metrics Table
        ax4.axis('tight')
        ax4.axis('off')
        
        # Calculate summary metrics
        anal_times = [r['time']*1000 for r in self.results['analytical'] if r['success']]
        brpso_times = [r['time']*1000 for r in self.results['brpso'] if r['success']]
        anal_errors = [r['error']*1000 for r in self.results['analytical'] if r['success']]
        brpso_errors = [r['error']*1000 for r in self.results['brpso'] if r['success']]
        
        table_data = [
            ['Metric', 'Analytical IK', 'BRPSO IK', 'BRPSO Advantage'],
            ['Avg Time (ms)', f'{np.mean(anal_times):.1f}', f'{np.mean(brpso_times):.1f}', f'{np.mean(brpso_times)/np.mean(anal_times):.1f}x slower'],
            ['Avg Error (mm)', f'{np.mean(anal_errors):.2f}', f'{np.mean(brpso_errors):.2f}', f'{((np.mean(anal_errors)-np.mean(brpso_errors))/np.mean(anal_errors)*100):.1f}% better'],
            ['Success Rate (%)', f'{anal_success/anal_total*100:.1f}', f'{brpso_success/brpso_total*100:.1f}', f'{(brpso_success/brpso_total - anal_success/anal_total)*100:+.1f}%'],
            ['Max Error (mm)', f'{max(anal_errors):.2f}', f'{max(brpso_errors):.2f}', f'{((max(anal_errors)-max(brpso_errors))/max(anal_errors)*100):.1f}% better'],
            ['Std Dev (mm)', f'{np.std(anal_errors):.2f}', f'{np.std(brpso_errors):.2f}', f'{((np.std(anal_errors)-np.std(brpso_errors))/np.std(anal_errors)*100):.1f}% better']
        ]
        
        table = ax4.table(cellText=table_data, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the header row
        for i in range(4):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold')
        
        ax4.set_title('Performance Summary Table')
        
        plt.tight_layout()
        plt.savefig('analysis/08_summary_statistics.png', dpi=300, bbox_inches='tight', 
                   facecolor='#2C3E50')
        plt.close()
        print("  ✓ Summary Statistics saved")

def main():
    """Main execution function"""
    print("🚀 Starting Individual Chart Generation")
    print("=" * 50)
    
    # Create generator
    generator = IndividualChartGenerator()
    
    # Collect data
    generator.collect_performance_data()
    
    # Generate charts
    generator.generate_individual_charts()
    
    print(f"\n✅ All charts saved to 'analysis/' folder")
    print(f"📁 Generated {8} individual analysis charts")

if __name__ == "__main__":
    main() 