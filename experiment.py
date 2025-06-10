import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from feddqn_agent import DQNAgent, FedAvgAggregator, CentralizedDQN
from baselines import pfs_scheduler, RoundRobinScheduler, static_average_scheduler, DDPGScheduler
from simulation_env import RANEnvironment
from config import *
import scipy.stats as st

class ExperimentRunner:
    def __init__(self):
        self.results_dir = 'results'
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.metrics = {
            'FedDQN': {'latency': [], 'fairness': [], 'throughput': [], 'comm_overhead': []},
            'C-DQN': {'latency': [], 'fairness': [], 'throughput': [], 'comm_overhead': []},
            'DDPG': {'latency': [], 'fairness': [], 'throughput': [], 'comm_overhead': []},
            'PFS': {'latency': [], 'fairness': [], 'throughput': [], 'comm_overhead': []},
            'RR': {'latency': [], 'fairness': [], 'throughput': [], 'comm_overhead': []},
            'SA': {'latency': [], 'fairness': [], 'throughput': [], 'comm_overhead': []}
        }
        
    def run_experiment(self, num_seeds=2, iterations_per_seed=100):
        for seed in range(num_seeds):
            print(f"\nRunning experiment with seed {seed}")
            np.random.seed(seed)
            
            env = RANEnvironment(NUM_BS, NUM_UE_PER_BS)
            state_shape = (NUM_UE_PER_BS, 2)
            
            fed_agents = [DQNAgent(state_shape, NUM_UE_PER_BS, i) for i in range(NUM_BS)]
            aggregator = FedAvgAggregator(fed_agents)
            c_dqn = CentralizedDQN((NUM_BS, NUM_UE_PER_BS, 2), NUM_UE_PER_BS, NUM_BS)
            ddpg = DDPGScheduler((NUM_BS, NUM_UE_PER_BS, 2), NUM_UE_PER_BS)
            rr_scheduler = RoundRobinScheduler(NUM_UE_PER_BS)
            
            for iteration in range(iterations_per_seed):
                state = env.get_state()
                
                actions = [agent.act(state[bs]) for bs, agent in enumerate(fed_agents)]
                next_state = env.step(actions)
                latency, fairness, throughput = env.get_metrics()
                self.metrics['FedDQN']['latency'].append(latency)
                self.metrics['FedDQN']['fairness'].append(fairness)
                self.metrics['FedDQN']['throughput'].append(throughput)
                self.metrics['FedDQN']['comm_overhead'].append(0.0)
                
                c_actions = c_dqn.act(state)
                env.step(c_actions)
                latency, fairness, throughput = env.get_metrics()
                self.metrics['C-DQN']['latency'].append(latency)
                self.metrics['C-DQN']['fairness'].append(fairness)
                self.metrics['C-DQN']['throughput'].append(throughput)
                self.metrics['C-DQN']['comm_overhead'].append(0.0)
                
                d_actions = ddpg.schedule(state)
                env.step(d_actions)
                latency, fairness, throughput = env.get_metrics()
                self.metrics['DDPG']['latency'].append(latency)
                self.metrics['DDPG']['fairness'].append(fairness)
                self.metrics['DDPG']['throughput'].append(throughput)
                self.metrics['DDPG']['comm_overhead'].append(0.0)
                
                pfs_actions = pfs_scheduler(env.channel_gains, 
                                          np.array([[len(q) for q in bs] for bs in env.queues]))
                env.step(pfs_actions)
                latency, fairness, throughput = env.get_metrics()
                self.metrics['PFS']['latency'].append(latency)
                self.metrics['PFS']['fairness'].append(fairness)
                self.metrics['PFS']['throughput'].append(throughput)
                self.metrics['PFS']['comm_overhead'].append(0.0)
                
                rr_actions = rr_scheduler.schedule()
                env.step(rr_actions)
                latency, fairness, throughput = env.get_metrics()
                self.metrics['RR']['latency'].append(latency)
                self.metrics['RR']['fairness'].append(fairness)
                self.metrics['RR']['throughput'].append(throughput)
                self.metrics['RR']['comm_overhead'].append(0.0)
                
                sa_actions = static_average_scheduler()
                env.step(sa_actions)
                latency, fairness, throughput = env.get_metrics()
                self.metrics['SA']['latency'].append(latency)
                self.metrics['SA']['fairness'].append(fairness)
                self.metrics['SA']['throughput'].append(throughput)
                self.metrics['SA']['comm_overhead'].append(0.0)
                
                for bs in range(NUM_BS):
                    fed_agents[bs].remember(
                        state[bs], 
                        actions[bs], 
                        0.0,
                        next_state[bs], 
                        False
                    )
                    fed_agents[bs].replay()
                
                if iteration % HYPERPARAMS["fedavg_interval"] == 0:
                    aggregator.aggregate()
                    weights_size = sum([w.size for w in fed_agents[0].model.get_weights()])
                    self.metrics['FedDQN']['comm_overhead'][-1] = weights_size * NUM_BS * 4 / 1e6
                
                if iteration % 10 == 0:
                    print(f"\rIteration {iteration}/{iterations_per_seed}", end="")
            
            print("\nExperiment completed!")
        
        self.plot_results()

    def plot_results(self):
        # Set up the style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = [14, 10]
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.titlesize'] = 24
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16
        plt.rcParams['legend.fontsize'] = 16
        
        # Process metrics for plotting
        algorithms = list(self.metrics.keys())
        metrics = ['latency', 'fairness', 'throughput', 'comm_overhead']
        
        # Calculate comprehensive statistics for each metric
        stats = {}
        for metric in metrics:
            means = []
            stds = []
            cis = []
            p_values = []
            data = []
            effect_sizes = []
            cohens_d = []
            
            for algo in algorithms:
                metric_data = np.array(self.metrics[algo][metric])
                means.append(np.mean(metric_data))
                stds.append(np.std(metric_data))
                
                # Calculate 95% confidence interval
                ci = st.t.interval(0.95, len(metric_data)-1,
                                 loc=np.mean(metric_data),
                                 scale=st.sem(metric_data))
                cis.append([means[-1] - ci[0], ci[1] - means[-1]])
                data.append(metric_data)
                
                # Calculate statistical significance and effect sizes
                if algo != 'FedDQN':
                    # T-test for statistical significance
                    t_stat, p_val = st.ttest_ind(
                        self.metrics['FedDQN'][metric],
                        self.metrics[algo][metric]
                    )
                    p_values.append(p_val)
                    
                    # Calculate Cohen's d effect size
                    pooled_std = np.sqrt((np.var(self.metrics['FedDQN'][metric]) + 
                                        np.var(self.metrics[algo][metric])) / 2)
                    d = (np.mean(self.metrics['FedDQN'][metric]) - 
                         np.mean(self.metrics[algo][metric])) / pooled_std
                    cohens_d.append(d)
                    
                    # Calculate effect size (percentage difference)
                    effect_size = ((np.mean(self.metrics['FedDQN'][metric]) - 
                                  np.mean(self.metrics[algo][metric])) / 
                                 np.mean(self.metrics[algo][metric])) * 100
                    effect_sizes.append(effect_size)
                else:
                    p_values.append(1.0)
                    cohens_d.append(0.0)
                    effect_sizes.append(0.0)
            
            stats[metric] = {
                'means': means,
                'stds': stds,
                'cis': cis,
                'p_values': p_values,
                'data': data,
                'labels': algorithms,
                'effect_sizes': effect_sizes,
                'cohens_d': cohens_d
            }

        # Plot each metric with enhanced statistical indicators
        for metric in metrics:
            plt.figure(figsize=(14, 10))
            data = stats[metric]
            x = np.arange(len(data['labels']))
            width = 0.7
            
            # Plot bars with error bars showing 95% confidence intervals
            bars = plt.bar(x, data['means'], width, yerr=np.array(data['cis']).T, capsize=15,
                          color=['#E63946', '#457B9D', '#1D3557', '#2A9D8F', '#E9C46A', '#F4A261'])
            
            # Add standard deviation as error bars
            plt.errorbar(x, data['means'], yerr=data['stds'], fmt='none', ecolor='black', 
                        capsize=5, capthick=1, alpha=0.5, label='Std Dev')
            
            # Add statistical annotations
            for i, (mean, std, ci, label, p_val, effect_size, cohens_d) in enumerate(
                zip(data['means'], data['stds'], data['cis'], 
                    data['labels'], data['p_values'],
                    data['effect_sizes'], data['cohens_d'])):
                
                # Statistical significance
                sig = '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                
                # Create detailed statistical annotation
                stats_text = (f'{sig}\n'
                             f'p={p_val:.3f}\n'
                             f'd={cohens_d:.2f}\n'
                             f'Δ={effect_size:.1f}%')
                
                plt.text(i, mean + ci[1] + 0.1, stats_text,
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Set title and labels based on metric
            metric_titles = {
                'latency': 'Average Latency Comparison with Statistical Analysis',
                'fairness': 'Fairness Index Comparison with Statistical Analysis',
                'throughput': 'Throughput Comparison with Statistical Analysis',
                'comm_overhead': 'Communication Overhead Comparison with Statistical Analysis'
            }
            metric_units = {
                'latency': 'Latency (ms)',
                'fairness': 'Fairness Index',
                'throughput': 'Throughput (Mbps)',
                'comm_overhead': 'Overhead (MB)'
            }
            
            plt.title(metric_titles[metric], pad=20)
            plt.xlabel('Algorithm', labelpad=15)
            plt.ylabel(metric_units[metric], labelpad=15)
            plt.xticks(x, data['labels'], rotation=45)
            
            # Add statistical summary
            summary_text = (
                f'Statistical Summary:\n'
                f'FedDQN: μ={data["means"][0]:.2f}, σ={data["stds"][0]:.2f}\n'
                f'95% CI: [{data["means"][0]-data["cis"][0][0]:.2f}, {data["means"][0]+data["cis"][0][1]:.2f}]'
            )
            plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add legend for error bars
            plt.legend(['95% CI', 'Std Dev'], loc='upper right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'{metric}_results.svg'), bbox_inches='tight', format='svg')
            plt.savefig(os.path.join(self.results_dir, f'{metric}_results.png'), bbox_inches='tight', dpi=300)
            plt.close()

        # Create radar plot with statistical indicators
        plt.figure(figsize=(14, 10))
        
        # Normalize metrics for radar plot
        normalized_means = {}
        normalized_stds = {}
        for metric in metrics:
            max_val = max(stats[metric]['means'])
            normalized_means[metric] = [m/max_val for m in stats[metric]['means']]
            normalized_stds[metric] = [s/max_val for s in stats[metric]['stds']]
        
        # Set up radar plot
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # close the loop
        
        # Plot each algorithm
        for i, algo in enumerate(algorithms):
            values = [normalized_means[metric][i] for metric in metrics]
            values = np.concatenate((values, [values[0]]))  # close the loop
            
            stds = [normalized_stds[metric][i] for metric in metrics]
            stds = np.concatenate((stds, [stds[0]]))  # close the loop
            
            plt.plot(angles, values, 'o-', linewidth=2, label=algo)
            plt.fill_between(angles, 
                           [v-s for v, s in zip(values, stds)],
                           [v+s for v, s in zip(values, stds)],
                           alpha=0.25)
        
        # Add labels and title
        plt.xticks(angles[:-1], metrics, size=12)
        plt.title('Performance Comparison with Standard Deviations', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Add statistical summary
        summary_text = 'Statistical Summary:\n'
        for metric in metrics:
            summary_text += f'\n{metric}:\n'
            for i, algo in enumerate(algorithms):
                summary_text += f'{algo}: μ={stats[metric]["means"][i]:.2f}, σ={stats[metric]["stds"][i]:.2f}\n'
        
        plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'algorithm_radar.svg'), bbox_inches='tight', format='svg')
        plt.savefig(os.path.join(self.results_dir, 'algorithm_radar.png'), bbox_inches='tight', dpi=300)
        plt.close()

        # Save statistics to CSV
        stats_df = pd.DataFrame()
        for algo in algorithms:
            for metric in metrics:
                metric_data = np.array(self.metrics[algo][metric])
                if len(metric_data) == 0:
                    stats_df.loc[algo, f'{metric}_mean'] = np.nan
                    stats_df.loc[algo, f'{metric}_std'] = np.nan
                    stats_df.loc[algo, f'{metric}_ci_low'] = np.nan
                    stats_df.loc[algo, f'{metric}_ci_high'] = np.nan
                else:
                    stats_df.loc[algo, f'{metric}_mean'] = np.mean(metric_data)
                    stats_df.loc[algo, f'{metric}_std'] = np.std(metric_data)
                    ci_low, ci_high = st.t.interval(
                        0.95, len(metric_data)-1,
                        loc=np.mean(metric_data),
                        scale=st.sem(metric_data)
                    )
                    stats_df.loc[algo, f'{metric}_ci_low'] = ci_low
                    stats_df.loc[algo, f'{metric}_ci_high'] = ci_high
        
        stats_df.to_csv(os.path.join(self.results_dir, 'experiment_stats.csv'))

if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run_experiment() 