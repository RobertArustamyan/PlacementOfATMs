import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from branch_bound_priority import BranchAndBoundSCPPriority, SearchStrategy, HeuristicStrategy, compare_strategies
import seaborn as sns

class EnhancedStrategyAnalyzer:
    """Comprehensive analyzer for branch-and-bound strategies with heuristics"""

    def __init__(self, test_files=None, output_dir="enhanced_results"):
        self.test_files = test_files or ['scp41.txt', 'scp42.txt', 'scp43.txt', 'scp44.txt']
        self.output_dir = output_dir

        # Search strategies to test
        self.search_strategies = [
            SearchStrategy.DEPTH_FIRST,
            SearchStrategy.BREADTH_FIRST,
            SearchStrategy.BEST_FIRST,
            SearchStrategy.HYBRID_DF_BF,
            SearchStrategy.MOST_FRACTIONAL
        ]

        # Heuristic strategies to test
        self.heuristic_strategies = [
            HeuristicStrategy.NONE,
            HeuristicStrategy.BASIC,
            HeuristicStrategy.INTERMEDIATE,
            HeuristicStrategy.ADVANCED
        ]

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def run_comprehensive_analysis(self, runs_per_test=5, time_limit=120):
        """Run comprehensive analysis of all strategy combinations"""
        print("Starting comprehensive strategy + heuristic analysis...")
        print(f"Test files: {self.test_files}")
        print(f"Search strategies: {[s.value for s in self.search_strategies]}")
        print(f"Heuristic strategies: {[h.value for h in self.heuristic_strategies]}")
        print(f"Runs per test: {runs_per_test}")
        print(f"Time limit per run: {time_limit}s")
        print("=" * 100)

        all_results = []

        for filename in self.test_files:
            print(f"\nTesting file: {filename}")
            print("-" * 70)

            for search_strategy in self.search_strategies:
                for heuristic_strategy in self.heuristic_strategies:
                    combo_name = f"{search_strategy.value}_{heuristic_strategy.value}"
                    print(f"Testing combination: {combo_name}")

                    combo_results = []

                    for run in range(runs_per_test):
                        try:
                            solver = BranchAndBoundSCPPriority(
                                filename, search_strategy, heuristic_strategy
                            )
                            result = solver.solve(
                                use_heuristic=(heuristic_strategy != HeuristicStrategy.NONE),
                                time_limit=time_limit
                            )

                            heuristic_info = result.getHeuristicInfo()
                            is_feasible = result.getOV() < float('inf')
                            is_optimal = result.isOptimal() if hasattr(result, 'isOptimal') else (
                                is_feasible and result.getGap() <= 1e-6 if hasattr(result, 'getGap') else False
                            )
                            run_data = {
                                'filename': filename,
                                'search_strategy': search_strategy.value,
                                'heuristic_strategy': heuristic_strategy.value,
                                'combination': combo_name,
                                'run': run + 1,
                                'cost': result.getOV(),
                                'time': result.getTime(),
                                'nodes_explored': result.getNodesExplored(),
                                'feasible': is_feasible,
                                'optimal': is_optimal,
                                'heuristic_time': heuristic_info.get('total_heuristic_time', 0),
                                'heuristic_calls': heuristic_info.get('heuristic_calls', 0),
                                'heuristic_improvements': heuristic_info.get('heuristic_improvements', 0),
                                'bb_time': result.getTime() - heuristic_info.get('total_heuristic_time', 0),
                                'heuristic_efficiency': (heuristic_info.get('heuristic_improvements', 0) /
                                                         max(heuristic_info.get('heuristic_calls', 1), 1))
                            }
                            combo_results.append(run_data)
                            all_results.append(run_data)

                            print(f"  Run {run + 1}: Cost={result.getOV():.0f}, "
                                  f"Time={result.getTime():.3f}s (BB: {run_data['bb_time']:.3f}s, "
                                  f"Heur: {heuristic_info.get('total_heuristic_time', 0):.3f}s), "
                                  f"Nodes={result.getNodesExplored()}, "
                                  f"HeurCalls={heuristic_info.get('heuristic_calls', 0)}, "
                                  f"HeurImpr={heuristic_info.get('heuristic_improvements', 0)}")

                        except Exception as e:
                            print(f"  Run {run + 1}: ERROR - {str(e)}")
                            error_data = {
                                'filename': filename,
                                'search_strategy': search_strategy.value,
                                'heuristic_strategy': heuristic_strategy.value,
                                'combination': combo_name,
                                'run': run + 1,
                                'cost': float('inf'),
                                'time': time_limit,
                                'nodes_explored': 0,
                                'optimal': False,
                                'heuristic_time': 0,
                                'heuristic_calls': 0,
                                'heuristic_improvements': 0,
                                'bb_time': time_limit,
                                'heuristic_efficiency': 0
                            }
                            all_results.append(error_data)

                    # Print combination summary
                    if combo_results:
                        successful_runs = [r for r in combo_results if r['optimal']]
                        if successful_runs:
                            avg_cost = np.mean([r['cost'] for r in successful_runs])
                            avg_time = np.mean([r['time'] for r in combo_results])
                            avg_bb_time = np.mean([r['bb_time'] for r in combo_results])
                            avg_heur_time = np.mean([r['heuristic_time'] for r in combo_results])
                            avg_nodes = np.mean([r['nodes_explored'] for r in combo_results])
                            success_rate = len(successful_runs) / len(combo_results)
                            avg_heur_efficiency = np.mean([r['heuristic_efficiency'] for r in combo_results])

                            print(f"  Summary: Success={success_rate:.2%}, Avg Cost={avg_cost:.1f}, "
                                  f"Avg Time={avg_time:.3f}s (BB: {avg_bb_time:.3f}s, Heur: {avg_heur_time:.3f}s), "
                                  f"Avg Nodes={avg_nodes:.0f}, Heur Efficiency={avg_heur_efficiency:.2f}")

        # Save results to CSV
        df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(self.output_dir, f"enhanced_strategy_results_{timestamp}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"\nResults saved to: {csv_filename}")

        return df

    def analyze_results(self, df):
        """Analyze and visualize the enhanced results"""
        print("\n" + "=" * 100)
        print("DETAILED ENHANCED ANALYSIS")
        print("=" * 100)

        # Overall statistics by combination
        summary_stats = df.groupby(['filename', 'combination']).agg({
            'cost': ['mean', 'min', 'std', 'count'],
            'time': ['mean', 'min', 'max', 'std'],
            'bb_time': ['mean'],
            'heuristic_time': ['mean'],
            'nodes_explored': ['mean', 'min', 'max'],
            'optimal': ['sum', 'count'],
            'heuristic_calls': ['mean'],
            'heuristic_improvements': ['mean'],
            'heuristic_efficiency': ['mean']
        }).round(4)

        print("\nSUMMARY STATISTICS BY COMBINATION:")
        print(summary_stats)

        # Success rates by combination
        success_rates = (
            df.groupby(['filename', 'combination'])['optimal']
            .agg(lambda x: x.sum() / len(x))
            .unstack(fill_value=0)
        )

        print("\nSUCCESS RATES BY COMBINATION:")
        print(success_rates.round(3))

        # Heuristic effectiveness analysis
        print("\nHEURISTIC EFFECTIVENESS ANALYSIS:")
        heuristic_analysis = df[df['heuristic_strategy'] != 'none'].groupby('heuristic_strategy').agg({
            'heuristic_time': 'mean',
            'heuristic_calls': 'mean',
            'heuristic_improvements': 'mean',
            'heuristic_efficiency': 'mean',
            'cost': 'mean',
            'time': 'mean'
        }).round(4)

        print(heuristic_analysis)

        # Best combination per file
        print("\nBEST COMBINATION PER FILE:")
        for filename in df['filename'].unique():
            file_data = df[(df['filename'] == filename) & (df['optimal'] == True)]
            if len(file_data) > 0:
                best_combo = file_data.groupby('combination')['cost'].mean().idxmin()
                best_cost = file_data.groupby('combination')['cost'].mean().min()
                combo_data = file_data[file_data['combination'] == best_combo]
                best_time = combo_data['time'].mean()
                best_bb_time = combo_data['bb_time'].mean()
                best_heur_time = combo_data['heuristic_time'].mean()

                print(f"  {filename}: {best_combo}")
                print(f"    Cost: {best_cost:.1f}, Total Time: {best_time:.3f}s "
                      f"(BB: {best_bb_time:.3f}s, Heur: {best_heur_time:.3f}s)")

        # Search strategy comparison (averaged across heuristic strategies)
        print("\nSEARCH STRATEGY COMPARISON (averaged across heuristic strategies):")
        search_comparison = df[df['optimal'] == True].groupby('search_strategy').agg({
            'cost': 'mean',
            'time': 'mean',
            'nodes_explored': 'mean'
        }).round(3)

        for strategy, stats in search_comparison.iterrows():
            print(f"  {strategy}: Avg Cost={stats['cost']:.1f}, "
                  f"Avg Time={stats['time']:.3f}s, Avg Nodes={stats['nodes_explored']:.0f}")

        # Heuristic strategy comparison (averaged across search strategies)
        print("\nHEURISTIC STRATEGY COMPARISON (averaged across search strategies):")
        heuristic_comparison = df[df['optimal'] == True].groupby('heuristic_strategy').agg({
            'cost': 'mean',
            'time': 'mean',
            'bb_time': 'mean',
            'heuristic_time': 'mean',
            'heuristic_calls': 'mean',
            'heuristic_improvements': 'mean'
        }).round(3)

        for heuristic, stats in heuristic_comparison.iterrows():
            print(f"  {heuristic}: Avg Cost={stats['cost']:.1f}, "
                  f"Total Time={stats['time']:.3f}s (BB: {stats['bb_time']:.3f}s, Heur: {stats['heuristic_time']:.3f}s), "
                  f"Heur Calls={stats['heuristic_calls']:.1f}, Improvements={stats['heuristic_improvements']:.1f}")


        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename_stats = os.path.join(self.output_dir, f"SummaryStats_{timestamp}.csv")
        csv_filename_rates = os.path.join(self.output_dir, f"SuccessRates_{timestamp}.csv")
        csv_filename_heursitic = os.path.join(self.output_dir, f"HeuristicAnalysis_{timestamp}.csv")

        summary_stats.to_csv(csv_filename_stats, index=True)
        success_rates.to_csv(csv_filename_rates, index=True)
        heuristic_analysis.to_csv(csv_filename_heursitic, index=True)

        return summary_stats, success_rates, heuristic_analysis

    def create_enhanced_visualizations(self, df):
        """Create enhanced visualizations for branch-and-bound strategy comparison"""
        plt.style.use('seaborn-v0_8')

        # Set up the plotting area
        fig, axes = plt.subplots(4, 2, figsize=(20, 24))
        fig.suptitle('Branch-and-Bound Strategy Comparison: Search Strategies vs Heuristics',
                     fontsize=16, fontweight='bold')

        # Filter successful runs (optimal solutions found)
        successful_runs = df[df['optimal'] == True].copy()

        # 1. Cost comparison by search strategy
        axes[0, 0].set_title('Solution Cost by Search Strategy', fontsize=12, fontweight='bold')
        if len(successful_runs) > 0:
            search_costs = successful_runs.groupby('search_strategy')['cost'].agg(['mean', 'std', 'count'])
            x_pos = range(len(search_costs))
            bars = axes[0, 0].bar(x_pos, search_costs['mean'],
                                  yerr=search_costs['std'], capsize=5, alpha=0.8)
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels(search_costs.index, rotation=0, ha='right')
            axes[0, 0].set_xlabel('Search Strategy')
            axes[0, 0].set_ylabel('Average Cost')

            # Add count labels on bars
            for i, (bar, count) in enumerate(zip(bars, search_costs['count'])):
                axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                                f'n={count}', ha='center', va='bottom', fontsize=9)

        # 2. Time comparison by search strategy
        axes[0, 1].set_title('Solution Time by Search Strategy', fontsize=12, fontweight='bold')
        if len(successful_runs) > 0:
            # Create box plot for time distribution
            search_strategies = successful_runs['search_strategy'].unique()
            time_data = [successful_runs[successful_runs['search_strategy'] == strategy]['time'].values
                         for strategy in search_strategies]

            box_plot = axes[0, 1].boxplot(time_data, labels=search_strategies, patch_artist=True)
            axes[0, 1].set_xlabel('Search Strategy')
            axes[0, 1].set_ylabel('Time (seconds)')
            axes[0, 1].tick_params(axis='x', rotation=0)

            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(box_plot['boxes'])))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)

        # 3. Nodes explored by search strategy
        axes[1, 0].set_title('Nodes Explored by Search Strategy', fontsize=12, fontweight='bold')
        if len(successful_runs) > 0:
            nodes_data = successful_runs.groupby('search_strategy')['nodes_explored'].agg(['mean', 'std'])
            x_pos = range(len(nodes_data))
            bars = axes[1, 0].bar(x_pos, nodes_data['mean'],
                                  yerr=nodes_data['std'], capsize=5, alpha=0.8, color='lightcoral')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(nodes_data.index, rotation=0, ha='right')
            axes[1, 0].set_xlabel('Search Strategy')
            axes[1, 0].set_ylabel('Average Nodes Explored')
            axes[1, 0].set_yscale('log')  # Use log scale due to potentially large differences

            # Add mean value labels on bars
            for i, (bar, mean_val) in enumerate(zip(bars, nodes_data['mean'])):
                axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                                f'{mean_val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 4. Heuristic impact analysis
        axes[1, 1].set_title('Impact of Heuristics on Performance', fontsize=12, fontweight='bold')
        if len(successful_runs) > 0:
            # Compare performance with and without heuristics
            no_heuristic = successful_runs[successful_runs['heuristic_strategy'] == 'none']
            with_heuristic = successful_runs[successful_runs['heuristic_strategy'] != 'none']

            if len(no_heuristic) > 0 and len(with_heuristic) > 0:
                categories = ['Avg Cost', 'Avg Time', 'Avg Nodes']
                no_heur_values = [
                    no_heuristic['cost'].mean(),
                    no_heuristic['time'].mean(),
                    no_heuristic['nodes_explored'].mean()
                ]
                with_heur_values = [
                    with_heuristic['cost'].mean(),
                    with_heuristic['time'].mean(),
                    with_heuristic['nodes_explored'].mean()
                ]

                x = np.arange(len(categories))
                width = 0.35

                bars1 = axes[1, 1].bar(x - width / 2, no_heur_values, width,
                                       label=f'No Heuristics (n={len(no_heuristic)})', alpha=0.8)
                bars2 = axes[1, 1].bar(x + width / 2, with_heur_values, width,
                                       label=f'With Heuristics (n={len(with_heuristic)})', alpha=0.8)

                axes[1, 1].set_xlabel('Performance Metrics')
                axes[1, 1].set_ylabel('Average Value')
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels(categories)
                axes[1, 1].legend()

        # 5. Success rate by combination with problem size categories (Stacked bar chart)
        axes[2, 0].set_title('Success Rate by Strategy: Small vs Large Problems', fontsize=12, fontweight='bold')

        # Create size categories
        df_with_size = df.copy()
        df_with_size['problem_size'] = df_with_size['num_sets'].apply(
            lambda x: 'Small (<2000 ATMs)' if x < 2000 else 'Large (≥2000 ATMs)')

        # Calculate success rates for each combination and size category
        success_by_size = df_with_size.groupby(['combination', 'problem_size']).agg({
            'optimal': ['sum', 'count']
        })
        success_by_size.columns = ['successes', 'total']
        success_by_size['success_rate'] = success_by_size['successes'] / success_by_size['total']
        success_by_size = success_by_size.reset_index()

        # Get top combinations by overall success rate
        overall_success = df_with_size.groupby('combination')['optimal'].mean().sort_values(ascending=False)
        top_10_combos = overall_success.head(10).index

        # Filter data for top combinations
        plot_data = success_by_size[success_by_size['combination'].isin(top_10_combos)]

        # Pivot for stacked bar chart
        pivot_data = plot_data.pivot(index='combination', columns='problem_size', values='success_rate').fillna(0)

        # Reorder by overall success rate
        pivot_data = pivot_data.reindex(top_10_combos)

        # Create stacked bar chart
        x_pos = range(len(pivot_data))
        if 'Small (<2000 ATMs)' in pivot_data.columns and 'Large (≥2000 ATMs)' in pivot_data.columns:
            bars1 = axes[2, 0].bar(x_pos, pivot_data['Small (<2000 ATMs)'],
                                   label='Small Problems (<2000 ATMs)', alpha=0.8, color='lightblue')
            bars2 = axes[2, 0].bar(x_pos, pivot_data['Large (≥2000 ATMs)'],
                                   bottom=pivot_data['Small (<2000 ATMs)'],
                                   label='Large Problems (≥2000 ATMs)', alpha=0.8, color='darkblue')
        elif 'Small (<2000 ATMs)' in pivot_data.columns:
            bars1 = axes[2, 0].bar(x_pos, pivot_data['Small (<2000 ATMs)'],
                                   label='Small Problems (<2000 ATMs)', alpha=0.8, color='lightblue')
        elif 'Large (≥2000 ATMs)' in pivot_data.columns:
            bars2 = axes[2, 0].bar(x_pos, pivot_data['Large (≥2000 ATMs)'],
                                   label='Large Problems (≥2000 ATMs)', alpha=0.8, color='darkblue')

        axes[2, 0].set_xticks(x_pos)
        axes[2, 0].set_xticklabels([combo.replace('_', '\n') for combo in pivot_data.index],
                                   rotation=90, ha='right', fontsize=9)
        axes[2, 0].set_xlabel('Strategy Combination')
        axes[2, 0].set_ylabel('Success Rate')
        axes[2, 0].set_ylim(0, 1.1)
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3, axis='y')

        # 6. Time vs Nodes Explored (Search Efficiency)
        axes[2, 1].set_title('Search Efficiency: Time vs Nodes Explored', fontsize=12, fontweight='bold')
        if len(successful_runs) > 0:
            strategies = successful_runs['search_strategy'].unique()
            colors = plt.cm.Set1(np.linspace(0, 1, len(strategies)))

            for strategy, color in zip(strategies, colors):
                strategy_data = successful_runs[successful_runs['search_strategy'] == strategy]
                axes[2, 1].scatter(strategy_data['time'], strategy_data['nodes_explored'],
                                   alpha=0.7, label=strategy, c=[color], s=50)

        axes[2, 1].set_xlabel('Time (seconds)')
        axes[2, 1].set_ylabel('Nodes Explored')
        axes[2, 1].set_yscale('log')  # Log scale for nodes
        axes[2, 1].legend(loc='lower right')
        axes[2, 1].grid(True, alpha=0.3)

        # 7. Problem Size Impact on Performance
        axes[3, 0].set_title('Performance vs Problem Size (Number of ATMs)', fontsize=12, fontweight='bold')
        if len(successful_runs) > 0:
            # Create scatter plot: problem size vs time, colored by search strategy
            strategies = successful_runs['search_strategy'].unique()
            colors = plt.cm.Set1(np.linspace(0, 1, len(strategies)))

            for strategy, color in zip(strategies, colors):
                strategy_data = successful_runs[successful_runs['search_strategy'] == strategy]
                axes[3, 0].scatter(strategy_data['num_sets'], strategy_data['time'],
                                   alpha=0.7, label=strategy, c=[color], s=50)

            axes[3, 0].set_xlabel('Number of ATMs (Sets)')
            axes[3, 0].set_ylabel('Time (seconds)')
            axes[3, 0].set_xscale('log')
            axes[3, 0].set_yscale('log')
            axes[3, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[3, 0].grid(True, alpha=0.3)

        # 8. Coverage Ratio Analysis
        axes[3, 1].set_title('Performance by Coverage Ratio (Users per ATM)', fontsize=12, fontweight='bold')
        if len(successful_runs) > 0:
            # Calculate coverage ratio
            successful_runs_copy = successful_runs.copy()
            successful_runs_copy['coverage_ratio'] = successful_runs_copy['num_users'] / successful_runs_copy[
                'num_sets']

            # Create bins for coverage ratio
            successful_runs_copy['ratio_category'] = pd.cut(successful_runs_copy['coverage_ratio'],
                                                            bins=5,
                                                            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

            # Calculate average time by coverage ratio category and search strategy
            ratio_time = successful_runs_copy.groupby(['ratio_category', 'search_strategy'])[
                'time'].mean().reset_index()

            # Create grouped bar chart
            categories = ratio_time['ratio_category'].unique()
            strategies = ratio_time['search_strategy'].unique()

            x = np.arange(len(categories))
            width = 0.15

            for i, strategy in enumerate(strategies):
                strategy_data = ratio_time[ratio_time['search_strategy'] == strategy]
                # Align categories properly
                values = []
                for cat in categories:
                    cat_data = strategy_data[strategy_data['ratio_category'] == cat]
                    if len(cat_data) > 0:
                        values.append(cat_data['time'].iloc[0])
                    else:
                        values.append(0)

                axes[3, 1].bar(x + i * width, values, width, label=strategy, alpha=0.8)

            axes[3, 1].set_xlabel('Coverage Ratio Category (Users/ATM)')
            axes[3, 1].set_ylabel('Average Time (seconds)')
            axes[3, 1].set_xticks(x + width * (len(strategies) - 1) / 2)
            axes[3, 1].set_xticklabels(categories, rotation=0)
            axes[3, 1].legend()
            axes[3, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Print summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(f"Total runs: {len(df)}")
        print(f"Successful runs: {len(successful_runs)} ({len(successful_runs) / len(df):.1%})")
        print(f"Search strategies tested: {df['search_strategy'].nunique()}")
        print(f"Heuristic strategies tested: {df['heuristic_strategy'].nunique()}")
        print(f"Small problems (<2000 ATMs): {len(df[df['num_sets'] < 2000])}")
        print(f"Large problems (≥2000 ATMs): {len(df[df['num_sets'] >= 2000])}")
        print(f"Average users per problem: {df['num_users'].mean():.1f}")
        print(f"Average ATMs per problem: {df['num_sets'].mean():.1f}")
        print(f"Average coverage ratio (users/ATM): {(df['num_users'] / df['num_sets']).mean():.2f}")

        if len(successful_runs) > 0:
            print(
                f"\nBest performing combination (lowest avg cost): {successful_runs.groupby('combination')['cost'].mean().idxmin()}")
            print(
                f"Fastest combination (lowest avg time): {successful_runs.groupby('combination')['time'].mean().idxmin()}")
            print(
                f"Most efficient nodes exploration: {successful_runs.groupby('combination')['nodes_explored'].mean().idxmin()}")

        # Save visualization with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(self.output_dir, f"bb_strategy_analysis_{timestamp}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\nVisualization saved to: {plot_filename}")
        print("=" * 60)

    def generate_enhanced_recommendations(self, df):
        """Generate enhanced recommendations based on comprehensive analysis"""
        print("\n" + "=" * 100)
        print("ENHANCED RECOMMENDATIONS")
        print("=" * 100)

        successful_runs = df[df['optimal'] == True]

        if len(successful_runs) == 0:
            print("No successful runs found. Consider increasing time limit or simplifying test cases.")
            return

        # 1. Overall best combination
        combo_performance = successful_runs.groupby('combination').agg({
            'time': 'mean',
            'cost': 'mean',
            'nodes_explored': 'mean',
            'heuristic_time': 'mean',
            'heuristic_efficiency': 'mean'
        })

        # Normalize metrics for combined scoring
        normalized_perf = combo_performance.copy()
        for col in ['time', 'cost', 'nodes_explored']:  # Lower is better
            if normalized_perf[col].max() > normalized_perf[col].min():
                normalized_perf[col] = (normalized_perf[col] - normalized_perf[col].min()) / \
                                       (normalized_perf[col].max() - normalized_perf[col].min())
            else:
                normalized_perf[col] = 0

        for col in ['heuristic_efficiency']:  # Higher is better
            if normalized_perf[col].max() > normalized_perf[col].min():
                normalized_perf[col] = 1 - (normalized_perf[col] - normalized_perf[col].min()) / \
                                       (normalized_perf[col].max() - normalized_perf[col].min())
            else:
                normalized_perf[col] = 0

        # Combined score
        normalized_perf['combined_score'] = (
                normalized_perf['time'] * 0.3 +
                normalized_perf['cost'] * 0.4 +
                normalized_perf['nodes_explored'] * 0.2 +
                normalized_perf['heuristic_efficiency'] * 0.1
        )

        best_overall = normalized_perf['combined_score'].idxmin()
        print(f"1. BEST OVERALL COMBINATION: {best_overall}")
        best_stats = combo_performance.loc[best_overall]
        print(f"   - Average cost: {best_stats['cost']:.1f}")
        print(f"   - Average time: {best_stats['time']:.3f}s")
        print(f"   - Average nodes: {best_stats['nodes_explored']:.0f}")
        print(f"   - Heuristic efficiency: {best_stats['heuristic_efficiency']:.3f}")

        # 2. Best search strategy (across all heuristic strategies)
        search_performance = successful_runs.groupby('search_strategy').agg({
            'time': 'mean',
            'cost': 'mean',
            'nodes_explored': 'mean'
        })

        fastest_search = search_performance['time'].idxmin()
        best_cost_search = search_performance['cost'].idxmin()
        most_efficient_search = search_performance['nodes_explored'].idxmin()

        print(f"\n2. SEARCH STRATEGY ANALYSIS:")
        print(f"   - Fastest: {fastest_search} ({search_performance.loc[fastest_search, 'time']:.3f}s avg)")
        print(f"   - Best cost: {best_cost_search} ({search_performance.loc[best_cost_search, 'cost']:.1f} avg)")
        print(
            f"   - Most efficient: {most_efficient_search} ({search_performance.loc[most_efficient_search, 'nodes_explored']:.0f} nodes avg)")

        # 3. Best heuristic strategy
        heuristic_performance = successful_runs.groupby('heuristic_strategy').agg({
            'cost': 'mean',
            'time': 'mean',
            'heuristic_time': 'mean',
            'heuristic_efficiency': 'mean',
            'heuristic_improvements': 'mean'
        })

        best_heuristic_cost = heuristic_performance['cost'].idxmin()
        most_efficient_heuristic = heuristic_performance['heuristic_efficiency'].idxmax()

        print(f"\n3. HEURISTIC STRATEGY ANALYSIS:")
        print(f"   - Best cost improvement: {best_heuristic_cost}")
        print(f"     * Average cost: {heuristic_performance.loc[best_heuristic_cost, 'cost']:.1f}")
        print(f"     * Heuristic overhead: {heuristic_performance.loc[best_heuristic_cost, 'heuristic_time']:.3f}s")
        print(f"   - Most efficient: {most_efficient_heuristic}")
        print(
            f"     * Efficiency ratio: {heuristic_performance.loc[most_efficient_heuristic, 'heuristic_efficiency']:.3f}")
        print(
            f"     * Average improvements: {heuristic_performance.loc[most_efficient_heuristic, 'heuristic_improvements']:.1f}")

        # 4. Problem size recommendations
        print(f"\n4. PROBLEM SIZE RECOMMENDATIONS:")
        for filename in df['filename'].unique():
            file_data = successful_runs[successful_runs['filename'] == filename]
            if len(file_data) > 0:
                best_combo_for_file = file_data.groupby('combination')['cost'].mean().idxmin()
                avg_time = file_data[file_data['combination'] == best_combo_for_file]['time'].mean()

                if avg_time < 10:
                    size_category = "Small"
                elif avg_time < 60:
                    size_category = "Medium"
                else:
                    size_category = "Large"

                print(f"   - {filename}: {size_category} problem")
                print(f"     * Best combination: {best_combo_for_file}")
                print(f"     * Average solve time: {avg_time:.1f}s")

        # 5. When to use heuristics
        print(f"\n5. HEURISTIC USAGE GUIDELINES:")

        # Compare performance with and without heuristics
        none_data = successful_runs[successful_runs['heuristic_strategy'] == 'none']
        heur_data = successful_runs[successful_runs['heuristic_strategy'] != 'none']

        if len(none_data) > 0 and len(heur_data) > 0:
            cost_improvement = (none_data['cost'].mean() - heur_data['cost'].mean()) / none_data['cost'].mean() * 100
            time_overhead = (heur_data['time'].mean() - none_data['time'].mean()) / none_data['time'].mean() * 100

            print(f"   - Average cost improvement with heuristics: {cost_improvement:.1f}%")
            print(f"   - Average time overhead with heuristics: {time_overhead:.1f}%")

            if cost_improvement > 5 and time_overhead < 50:
                print(f"   - RECOMMENDATION: Use heuristics - significant cost improvement with reasonable overhead")
            elif cost_improvement > 0 and time_overhead < 20:
                print(f"   - RECOMMENDATION: Use heuristics - modest improvement with low overhead")
            else:
                print(f"   - RECOMMENDATION: Consider problem-specific analysis before using heuristics")

        # 6. Specific strategy insights
        print(f"\n6. STRATEGY-SPECIFIC INSIGHTS:")
        print(f"   SEARCH STRATEGIES:")
        print(f"   - Depth-First: Fast initial solutions, good for finding feasible solutions quickly")
        print(f"   - Breadth-First: More systematic exploration, better bounds")
        print(f"   - Best-First: Balanced approach, good overall performance")
        print(f"   - Hybrid: Combines benefits, adapts during search")
        print(f"   - Most-Fractional: Focuses on difficult branching decisions")

        print(f"\n   HEURISTIC STRATEGIES:")
        print(f"   - None: Fastest for small problems, baseline comparison")
        print(f"   - Basic: Quick constructive heuristics, minimal overhead")
        print(f"   - Intermediate: Adds local search, good balance of improvement vs time")
        print(f"   - Advanced: Full adaptive approach, best for complex problems")

        # 7. Final recommendations by problem type
        print(f"\n7. FINAL RECOMMENDATIONS BY PROBLEM TYPE:")

        small_problems = df[df['time'] < 10]['filename'].unique()
        medium_problems = df[(df['time'] >= 10) & (df['time'] < 60)]['filename'].unique()
        large_problems = df[df['time'] >= 60]['filename'].unique()

        print(f"   - SMALL PROBLEMS ({len(small_problems)} files): Use DEPTH_FIRST + BASIC heuristics")
        print(f"   - MEDIUM PROBLEMS ({len(medium_problems)} files): Use BEST_FIRST + INTERMEDIATE heuristics")
        print(f"   - LARGE PROBLEMS ({len(large_problems)} files): Use HYBRID_DF_BF + ADVANCED heuristics")


def main():
    """Main function to run the complete analysis"""
    print("Enhanced Branch-and-Bound Strategy Analysis with Heuristics")
    print("=" * 70)

    # Initialize analyzer

    analyzer = EnhancedStrategyAnalyzer(
        test_files=[f for f in os.listdir('../SCPData') if f.endswith('.txt')],
        output_dir="enhanced_strategy_analysis_results"
    )

    # Run comprehensive analysis (If code is already executed comment the lower one and uncomment the second line)
    df = analyzer.run_comprehensive_analysis(runs_per_test=1, time_limit=120)
    # df = pd.read_csv("enhanced_strategy_analysis_results/combined_results_2.csv")

    # Analyze results
    summary_stats, success_rates, heuristic_analysis = analyzer.analyze_results(df)

    # Create enhanced visualizations
    analyzer.create_enhanced_visualizations(df)

    # Generate enhanced recommendations
    analyzer.generate_enhanced_recommendations(df)

    print(f"\nEnhanced analysis complete! Results saved in: {analyzer.output_dir}")

    # Save detailed analysis report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = os.path.join(analyzer.output_dir, f"analysis_report_{timestamp}.txt")

    # This would save the printed output to a file for later reference
    print(f"Consider redirecting output to file: {report_filename}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()