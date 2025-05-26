import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
from branch_bound_priority import BranchAndBoundSCPPriority, SearchStrategy, compare_strategies


class StrategyAnalyzer:
    """Comprehensive analyzer for branch-and-bound strategies"""

    def __init__(self, test_files=None, output_dir="results"):
        self.test_files = test_files or ['scp41.txt', 'scp42.txt', 'scp43.txt', 'scp44.txt']
        self.output_dir = output_dir
        self.strategies = [
            SearchStrategy.DEPTH_FIRST,
            SearchStrategy.BREADTH_FIRST,
            SearchStrategy.BEST_FIRST,
            SearchStrategy.HYBRID_DF_BF,
            SearchStrategy.MOST_FRACTIONAL
        ]

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def run_comprehensive_analysis(self, runs_per_test=5, time_limit=120):
        """Run comprehensive analysis of all strategies on all test files"""
        print("Starting comprehensive strategy analysis...")
        print(f"Test files: {self.test_files}")
        print(f"Strategies: {[s.value for s in self.strategies]}")
        print(f"Runs per test: {runs_per_test}")
        print(f"Time limit per run: {time_limit}s")
        print("=" * 80)

        all_results = []

        for filename in self.test_files:
            print(f"\nTesting file: {filename}")
            print("-" * 50)

            for strategy in self.strategies:
                print(f"Testing strategy: {strategy.value}")

                strategy_results = []

                for run in range(runs_per_test):
                    try:
                        solver = BranchAndBoundSCPPriority(filename, strategy)
                        result = solver.solve(use_heuristic=True, time_limit=time_limit)

                        run_data = {
                            'filename': filename,
                            'strategy': strategy.value,
                            'run': run + 1,
                            'cost': result.getOV(),
                            'time': result.getTime(),
                            'nodes_explored': result.getNodesExplored(),
                            'optimal': result.getOV() < float('inf')
                        }
                        strategy_results.append(run_data)
                        all_results.append(run_data)

                        print(
                            f"  Run {run + 1}: Cost={result.getOV():.0f}, Time={result.getTime():.3f}s, Nodes={result.getNodesExplored()}")

                    except Exception as e:
                        print(f"  Run {run + 1}: ERROR - {str(e)}")
                        error_data = {
                            'filename': filename,
                            'strategy': strategy.value,
                            'run': run + 1,
                            'cost': float('inf'),
                            'time': time_limit,
                            'nodes_explored': 0,
                            'optimal': False
                        }
                        all_results.append(error_data)

                # Print strategy summary
                if strategy_results:
                    avg_cost = np.mean([r['cost'] for r in strategy_results if r['optimal']])
                    avg_time = np.mean([r['time'] for r in strategy_results])
                    avg_nodes = np.mean([r['nodes_explored'] for r in strategy_results])
                    success_rate = sum(r['optimal'] for r in strategy_results) / len(strategy_results)

                    print(f"  Summary: Avg Cost={avg_cost:.1f}, Avg Time={avg_time:.3f}s, "
                          f"Avg Nodes={avg_nodes:.0f}, Success Rate={success_rate:.2%}")

        # Save results to CSV
        df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(self.output_dir, f"strategy_results_{timestamp}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"\nResults saved to: {csv_filename}")

        return df

    def analyze_results(self, df):
        """Analyze and visualize the results"""
        print("\n" + "=" * 80)
        print("DETAILED ANALYSIS")
        print("=" * 80)

        # Overall statistics
        summary_stats = df.groupby(['filename', 'strategy']).agg({
            'cost': ['mean', 'min', 'std', 'count'],
            'time': ['mean', 'min', 'max', 'std'],
            'nodes_explored': ['mean', 'min', 'max'],
            'optimal': ['sum', 'count']
        }).round(4)

        print("\nSUMMARY STATISTICS:")
        print(summary_stats)

        # Success rates
        # success_rates = df.groupby(['filename', 'strategy']).apply(
        #     lambda x: x['optimal'].sum() / len(x)
        # ).unstack(fill_value=0)

        success_rates = (
            df.groupby(['filename', 'strategy'])['optimal']
            .agg(lambda x: x.sum() / len(x))
            .unstack(fill_value=0)
        )

        print("\nSUCCESS RATES (fraction of runs finding optimal solution):")
        print(success_rates.round(3))

        # Best strategies per file
        print("\nBEST STRATEGY PER FILE (by average cost of successful runs):")
        for filename in df['filename'].unique():
            file_data = df[(df['filename'] == filename) & (df['optimal'] == True)]
            if len(file_data) > 0:
                best_strategy = file_data.groupby('strategy')['cost'].mean().idxmin()
                best_cost = file_data.groupby('strategy')['cost'].mean().min()
                best_time = file_data[file_data['strategy'] == best_strategy]['time'].mean()
                print(f"  {filename}: {best_strategy} (Cost: {best_cost:.1f}, Avg Time: {best_time:.3f}s)")

        # Speed comparison
        print("\nSPEED COMPARISON (average time for successful runs):")
        speed_comparison = df[df['optimal'] == True].groupby('strategy')['time'].mean().sort_values()
        for strategy, avg_time in speed_comparison.items():
            print(f"  {strategy}: {avg_time:.3f}s")

        return summary_stats, success_rates

    def create_visualizations(self, df):
        """Create visualizations of the results"""
        plt.style.use('seaborn-v0_8')

        # Set up the plotting area
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Branch-and-Bound Strategy Comparison', fontsize=16, fontweight='bold')

        # 1. Time comparison boxplot
        successful_runs = df[df['optimal'] == True]
        if len(successful_runs) > 0:
            axes[0, 0].set_title('Solution Time Distribution (Successful Runs)')
            successful_runs.boxplot(column='time', by='strategy', ax=axes[0, 0])
            axes[0, 0].set_xlabel('Strategy')
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. Nodes explored comparison
        if len(successful_runs) > 0:
            axes[0, 1].set_title('Nodes Explored Distribution')
            successful_runs.boxplot(column='nodes_explored', by='strategy', ax=axes[0, 1])
            axes[0, 1].set_xlabel('Strategy')
            axes[0, 1].set_ylabel('Nodes Explored')
            axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. Success rate by strategy
        # success_rates = df.groupby('strategy').apply(lambda x: x['optimal'].sum() / len(x))
        success_rates = df.groupby('strategy')['optimal'].agg(lambda x: x.sum() / len(x))

        axes[1, 0].bar(success_rates.index, success_rates.values)
        axes[1, 0].set_title('Success Rate by Strategy')
        axes[1, 0].set_xlabel('Strategy')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_ylim(0, 1)

        # 4. Average cost by strategy (successful runs only)
        if len(successful_runs) > 0:
            avg_costs = successful_runs.groupby('strategy')['cost'].mean()
            axes[1, 1].bar(avg_costs.index, avg_costs.values)
            axes[1, 1].set_title('Average Solution Cost (Successful Runs)')
            axes[1, 1].set_xlabel('Strategy')
            axes[1, 1].set_ylabel('Average Cost')
            axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(self.output_dir, f"strategy_comparison_{timestamp}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Visualization saved to: {plot_filename}")

    def generate_recommendations(self, df):
        """Generate recommendations based on analysis"""
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        successful_runs = df[df['optimal'] == True]

        if len(successful_runs) == 0:
            print("No successful runs found. Consider increasing time limit or simplifying test cases.")
            return

        # Overall best strategy
        strategy_performance = successful_runs.groupby('strategy').agg({
            'time': 'mean',
            'cost': 'mean',
            'nodes_explored': 'mean'
        })

        # Normalize metrics (lower is better for all)
        normalized_perf = strategy_performance.copy()
        for col in normalized_perf.columns:
            normalized_perf[col] = (normalized_perf[col] - normalized_perf[col].min()) / \
                                   (normalized_perf[col].max() - normalized_perf[col].min())

        # Combined score (equal weight to all metrics)
        normalized_perf['combined_score'] = normalized_perf.mean(axis=1)
        best_overall = normalized_perf['combined_score'].idxmin()

        print(f"1. BEST OVERALL STRATEGY: {best_overall}")
        print(f"   - Balanced performance across time, cost, and node exploration")

        # Fastest strategy
        fastest = strategy_performance['time'].idxmin()
        print(f"\n2. FASTEST STRATEGY: {fastest}")
        print(f"   - Average time: {strategy_performance.loc[fastest, 'time']:.3f}s")

        # Most efficient (fewest nodes)
        most_efficient = strategy_performance['nodes_explored'].idxmin()
        print(f"\n3. MOST EFFICIENT STRATEGY (fewest nodes): {most_efficient}")
        print(f"   - Average nodes explored: {strategy_performance.loc[most_efficient, 'nodes_explored']:.0f}")

        # Success rate analysis
        # success_rates = df.groupby('strategy').apply(lambda x: x['optimal'].sum() / len(x))
        success_rates = df.groupby('strategy')['optimal'].agg(lambda x: x.sum() / len(x))

        most_reliable = success_rates.idxmax()
        print(f"\n4. MOST RELIABLE STRATEGY: {most_reliable}")
        print(f"   - Success rate: {success_rates[most_reliable]:.2%}")

        print(f"\n5. STRATEGY INSIGHTS:")
        print(f"   - Depth-First: Good for finding solutions quickly but may not be optimal")
        print(f"   - Breadth-First: More systematic but can be slower")
        print(f"   - Best-First: Balances exploration with promising directions")
        print(f"   - Hybrid: Combines benefits of multiple approaches")
        print(f"   - Most-Fractional: Focuses on challenging branching decisions")


def main():
    """Main function to run the complete analysis"""
    print("Branch-and-Bound Strategy Analysis")
    print("=" * 50)

    # Initialize analyzer
    analyzer = StrategyAnalyzer(
        test_files=[f for f in os.listdir('../SCPData') if f.endswith('.txt')],  # Start with one file for testing
        output_dir="strategy_analysis_results"
    )

    # Run analysis
    df = analyzer.run_comprehensive_analysis(runs_per_test=3, time_limit=60)

    # Analyze results
    summary_stats, success_rates = analyzer.analyze_results(df)

    # Create visualizations
    analyzer.create_visualizations(df)

    # Generate recommendations
    analyzer.generate_recommendations(df)

    print(f"\nAnalysis complete! Results saved in: {analyzer.output_dir}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()