import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from feddqn_agent import DQNAgent, FedAvgAggregator, CentralizedDQN
from baselines import pfs_scheduler, RoundRobinScheduler, static_average_scheduler, DDPGScheduler
from simulation_env import RANEnvironment
from config import *

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
        
    def run_experiment(self, num_seeds=2, iterations_per_seed=50):
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
        import matplotlib as mpl
        import scipy.stats as st
        
        feddqn_color = '#FF6B6B'
        baseline_colors = {
            'C-DQN': '#4ECDC4',
            'DDPG': '#45B7D1',
            'PFS': '#96CEB4',
            'RR': '#FFEEAD',
            'SA': '#D4A5A5'
        }

        plt.style.use('default')
        sns.set_style("whitegrid")
        mpl.rcParams['figure.figsize'] = [12, 8]
        mpl.rcParams['axes.labelweight'] = 'bold'
        mpl.rcParams['axes.titleweight'] = 'bold'
        mpl.rcParams['axes.titlesize'] = 22
        mpl.rcParams['axes.labelsize'] = 18
        mpl.rcParams['xtick.labelsize'] = 16
        mpl.rcParams['ytick.labelsize'] = 16
        mpl.rcParams['legend.fontsize'] = 14
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        mpl.rcParams['axes.edgecolor'] = '#2C3E50'
        mpl.rcParams['axes.linewidth'] = 1.5
        mpl.rcParams['xtick.major.width'] = 1.5
        mpl.rcParams['ytick.major.width'] = 1.5
        mpl.rcParams['axes.grid'] = True
        mpl.rcParams['grid.alpha'] = 0.3
        mpl.rcParams['grid.color'] = '#95A5A6'
        mpl.rcParams['axes.facecolor'] = '#F8F9FA'
        mpl.rcParams['figure.facecolor'] = 'white'

        all_data = {}
        stats_rows = []
        for metric in ['latency', 'fairness', 'throughput', 'comm_overhead']:
            data = []
            labels = []
            means = []
            stds = []
            cis = []
            p_values = []
            
            for name, values in self.metrics.items():
                arr = np.array(values[metric])
                data.append(arr)
                labels.append(name)
                mean = np.mean(arr)
                std = np.std(arr)
                n = len(arr)
                ci = st.t.interval(0.95, n-1, loc=mean, scale=std/np.sqrt(n)) if n > 1 else (mean, mean)
                means.append(mean)
                stds.append(std)
                cis.append((ci[1]-mean, mean-ci[0]))
                
                if name != 'FedDQN':
                    t_stat, p_val = st.ttest_ind(data[0], arr, equal_var=False)
                    p_values.append(p_val)
                else:
                    p_values.append(1.0)
                
                if name != 'FedDQN' and std > 0:
                    effect_size = (mean - means[0]) / std
                else:
                    effect_size = 0
                
                stats_rows.append({
                    'algorithm': name,
                    'metric': metric,
                    'mean': mean,
                    'std': std,
                    'ci_low': ci[0],
                    'ci_high': ci[1],
                    'min': np.min(arr),
                    'max': np.max(arr),
                    'median': np.median(arr),
                    'iqr': st.iqr(arr),
                    'n_samples': n,
                    'p_value': p_values[-1],
                    'effect_size': effect_size
                })
            
            all_data[metric] = {
                'data': data,
                'labels': labels,
                'means': means,
                'stds': stds,
                'cis': cis,
                'p_values': p_values
            }

        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor=feddqn_color, edgecolor='black', label='FedDQN', linewidth=2, hatch='////'),
            plt.Rectangle((0,0),1,1, facecolor=baseline_colors['C-DQN'], edgecolor='black', label='C-DQN', linewidth=1.5),
            plt.Rectangle((0,0),1,1, facecolor=baseline_colors['DDPG'], edgecolor='black', label='DDPG', linewidth=1.5),
            plt.Rectangle((0,0),1,1, facecolor=baseline_colors['PFS'], edgecolor='black', label='PFS', linewidth=1.5),
            plt.Rectangle((0,0),1,1, facecolor=baseline_colors['RR'], edgecolor='black', label='RR', linewidth=1.5),
            plt.Rectangle((0,0),1,1, facecolor=baseline_colors['SA'], edgecolor='black', label='SA', linewidth=1.5)
        ]

        plt.figure(figsize=(12, 8))
        data = all_data['latency']
        x = np.arange(len(data['labels']))
        width = 0.7
        
        plt.gca().set_facecolor('#F8F9FA')
        plt.grid(True, linestyle='--', alpha=0.3, color='#95A5A6')
        
        bar_colors = [feddqn_color] + [baseline_colors[label] for label in data['labels'][1:]]
        bar_hatches = ['////'] + [''] * (len(data['labels'])-1)
        bars = plt.bar(x, data['means'], width, yerr=np.array(data['cis']).T, capsize=10,
                      color=bar_colors, edgecolor='#2C3E50', linewidth=2, alpha=0.95)
        
        for bar, hatch in zip(bars, bar_hatches):
            bar.set_hatch(hatch)
            if hatch == '////':
                bar.set_linewidth(2.5)
                bar.set_alpha(0.85)
            else:
                bar.set_linewidth(2)
                bar.set_alpha(0.6)
        
        feddqn_mean = data['means'][0]
        feddqn_arr = data['data'][0]
        for i, (mean, arr, label, p_val) in enumerate(zip(data['means'][1:], data['data'][1:], data['labels'][1:], data['p_values'][1:]), 1):
            sig = '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            pooled_std = np.sqrt((np.var(feddqn_arr) + np.var(arr)) / 2)
            effect_size = (mean - feddqn_mean) / pooled_std if pooled_std > 0 else 0
            improvement = ((mean - feddqn_mean) / abs(mean)) * 100 if mean != 0 else 0
            
            annotation = f'↑{improvement:.1f}%\n{sig}\nd={effect_size:.2f}'
            plt.text(i, mean + data['cis'][i][0] + 0.5, annotation,
                    ha='center', va='bottom', color='#2C3E50', fontweight='bold', fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.95, edgecolor='#95A5A6', 
                            pad=2, boxstyle='round,pad=0.5'))
            
            plt.plot([0, i], [feddqn_mean, mean], 'k--', alpha=0.7, linewidth=2)
            if p_val < 0.05:
                plt.plot([0, i], [feddqn_mean + data['cis'][0][0], mean + data['cis'][i][0] + 0.5],
                        color='#FF6B6B', alpha=0.5, linewidth=2)
        
        plt.text(0, feddqn_mean + data['cis'][0][0] + 0.5, 
                f"{feddqn_mean:.1f}±{data['stds'][0]:.1f}",
                ha='center', va='bottom', color='#2C3E50', fontweight='bold', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.95, edgecolor='#95A5A6', 
                        pad=2, boxstyle='round,pad=0.5'))
        
        plt.title('Average Latency (ms)\nFedDQN vs Baseline Algorithms', 
                 fontsize=22, pad=20, color='#2C3E50')
        plt.ylabel('Latency (ms)', fontsize=18, labelpad=10, color='#2C3E50')
        plt.xticks(x, data['labels'], rotation=30, ha='right', fontsize=16, color='#2C3E50')
        plt.yticks(fontsize=16, color='#2C3E50')
        
        plt.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=14,
                  framealpha=0.95, edgecolor='#95A5A6', facecolor='white',
                  bbox_to_anchor=(1.15, 1.0))
        
        plt.figtext(0.02, 0.02, 
                   f'Statistical Summary:\n* p < 0.05, ** p < 0.01\nError bars: 95% CI\nd: Cohen\'s effect size\nMean ± Std Dev: {feddqn_mean:.2f} ± {data["stds"][0]:.2f}',
                   fontsize=12, color='#2C3E50',
                   bbox=dict(facecolor='white', alpha=0.95, edgecolor='#95A5A6', 
                           pad=2, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'latency_results.png'), 
                   bbox_inches='tight', dpi=400, facecolor='white')
        plt.savefig(os.path.join(self.results_dir, 'latency_results.pdf'), 
                   bbox_inches='tight', dpi=600, facecolor='white')
        plt.close()

        plt.figure(figsize=(12, 8))
        data = all_data['fairness']
        x = np.arange(len(data['labels']))
        plt.gca().set_facecolor('#f8f9fa')
        plt.grid(True, linestyle='--', alpha=0.3)
        for i, (mean, std, label) in enumerate(zip(data['means'], data['stds'], data['labels'])):
            color = feddqn_color if label == 'FedDQN' else baseline_colors[label]
            marker = 'o'
            plt.plot(x[i], mean, marker+'-', color=color, linewidth=3 if label=='FedDQN' else 2, label=label, markersize=10 if label=='FedDQN' else 8, markerfacecolor='white')
            plt.fill_between([x[i]-0.1, x[i]+0.1], [mean-data['cis'][i][0], mean-data['cis'][i][0]], [mean+data['cis'][i][1], mean+data['cis'][i][1]], color=color, alpha=0.2)
        for i, (mean, std, label, p_val) in enumerate(zip(data['means'][1:], data['stds'][1:], data['labels'][1:], data['p_values'][1:]), 1):
            sig = '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            pooled_std = np.sqrt((np.var(data['data'][0]) + np.var(data['data'][i])) / 2)
            effect_size = (mean - data['means'][0]) / pooled_std if pooled_std > 0 else 0
            plt.text(x[i], mean + data['cis'][i][1] + 0.01, f'{sig}\nd={effect_size:.2f}', ha='center', va='bottom', fontsize=12, color='black', fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=2))
        plt.title("Jain's Fairness Index\nFedDQN vs Baseline Algorithms", fontsize=22, pad=20)
        plt.ylabel("Fairness Index", fontsize=18, labelpad=10)
        plt.xticks(x, data['labels'], rotation=30, ha='right', fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=14, framealpha=0.9, edgecolor='gray')
        plt.figtext(0.02, 0.02, 
                   f'Statistical Summary:\n* p < 0.05, ** p < 0.01\nShaded areas: 95% CI\nd: Cohen\'s effect size\nMean ± Std Dev: {mean:.2f} ± {std:.2f}',
                   fontsize=12, color='#2C3E50',
                   bbox=dict(facecolor='white', alpha=0.95, edgecolor='#95A5A6', 
                           pad=2, boxstyle='round,pad=0.5'))
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'fairness_results.png'), bbox_inches='tight', dpi=400)
        plt.savefig(os.path.join(self.results_dir, 'fairness_results.pdf'), bbox_inches='tight', dpi=600)
        plt.close()

        plt.figure(figsize=(14, 8))
        data = all_data['throughput']
        
        labels = ['FedDQN'] + [label for label in data['labels'] if label != 'FedDQN']
        sorted_data = [data['data'][data['labels'].index(label)] for label in labels]
        sorted_p_values = [data['p_values'][data['labels'].index(label)] for label in labels]
        
        plt.gca().set_facecolor('#F8F9FA')
        plt.grid(True, linestyle='--', alpha=0.3, color='#95A5A6')
        
        df = pd.DataFrame({
            'Algorithm': np.repeat(labels, [len(arr) for arr in sorted_data]),
            'Throughput': np.concatenate(sorted_data)
        })
        
        palette = {label: (feddqn_color if label=='FedDQN' else baseline_colors[label]) for label in labels}
        
        ax = sns.boxenplot(data=df, y='Algorithm', x='Throughput', hue='Algorithm', 
                         palette=palette, linewidth=2, k_depth='full', 
                         showfliers=True, saturation=0.9, dodge=False)
        
        n_labels = len(labels)
        for i, box in enumerate(ax.collections):
            if i >= n_labels:
                break
            label = labels[i]
            if label == 'FedDQN':
                box.set_alpha(0.85)
                box.set_linewidth(2.5)
                box.set_hatch('////')
                box.set_edgecolor('#2C3E50')
            else:
                box.set_alpha(0.6)
                box.set_linewidth(2)
                box.set_edgecolor('#2C3E50')
        
        for i, (arr, label, p_val) in enumerate(zip(sorted_data, labels, sorted_p_values)):
            mean = np.mean(arr)
            std = np.std(arr)
            if label != 'FedDQN':
                pooled_std = np.sqrt((np.var(sorted_data[0]) + np.var(arr)) / 2)
                effect_size = (mean - np.mean(sorted_data[0])) / pooled_std if pooled_std > 0 else 0
                sig = '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                annotation = f"{mean:.1f}±{std:.1f}\n{sig}\nd={effect_size:.2f}"
            else:
                annotation = f"{mean:.1f}±{std:.1f}"
            plt.text(mean, i, annotation, va='center', ha='left', fontsize=12, 
                    color='#2C3E50', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.95, edgecolor='#95A5A6', 
                            pad=2, boxstyle='round,pad=0.5'))
        
        plt.title('Throughput Distribution (Mbps)\nFedDQN vs Baseline Algorithms', 
                 fontsize=22, pad=20, color='#2C3E50')
        plt.xlabel('Throughput (Mbps)', fontsize=18, labelpad=10, color='#2C3E50')
        plt.ylabel('Algorithm', fontsize=18, labelpad=10, color='#2C3E50')
        plt.xticks(fontsize=16, color='#2C3E50')
        plt.yticks(fontsize=16, color='#2C3E50')
        
        plt.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=14,
                  framealpha=0.95, edgecolor='#95A5A6', facecolor='white',
                  bbox_to_anchor=(1.15, 1.0))
        
        plt.figtext(0.02, 0.02, 
                   f'Statistical Summary:\n* p < 0.05, ** p < 0.01\nBoxes: IQR, Whiskers: 1.5×IQR\nd: Cohen\'s effect size\nMean ± Std Dev: {mean:.2f} ± {std:.2f}',
                   fontsize=12, color='#2C3E50',
                   bbox=dict(facecolor='white', alpha=0.95, edgecolor='#95A5A6', 
                           pad=2, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'throughput_results.png'), 
                   bbox_inches='tight', dpi=400, facecolor='white')
        plt.savefig(os.path.join(self.results_dir, 'throughput_results.pdf'), 
                   bbox_inches='tight', dpi=600, facecolor='white')
        plt.close()

        plt.figure(figsize=(12, 8))
        data = all_data['comm_overhead']
        
        plt.gca().set_facecolor('#F8F9FA')
        plt.grid(True, linestyle='--', alpha=0.3, color='#95A5A6')
        
        plot_data = []
        labels = ['FedDQN'] + [label for label in data['labels'] if label != 'FedDQN']
        ordered_data = [data['data'][data['labels'].index(label)] for label in labels]
        ordered_p_values = [data['p_values'][data['labels'].index(label)] for label in labels]
        
        for label, arr in zip(labels, ordered_data):
            plot_data.extend([(x, label) for x in arr])
        df = pd.DataFrame(plot_data, columns=['Overhead', 'Algorithm'])
        
        ax = sns.violinplot(data=df, x='Algorithm', y='Overhead', 
                          inner='box', linewidth=2.5, saturation=0.95, cut=0, width=0.8)
        
        colors = {
            'FedDQN': '#FF6B6B',
            'C-DQN': '#4ECDC4',
            'DDPG': '#45B7D1',
            'PFS': '#96CEB4',
            'RR': '#FFEEAD',
            'SA': '#D4A5A5'
        }
        
        for i, patch in enumerate(ax.patches):
            if i < len(labels):
                label = labels[i]
                if label == 'FedDQN':
                    patch.set_alpha(0.85)
                    patch.set_linewidth(3)
                    patch.set_hatch('////')
                    patch.set_edgecolor('#2C3E50')
                    patch.set_facecolor(feddqn_color)
                else:
                    patch.set_alpha(0.6)
                    patch.set_linewidth(2)
                    patch.set_edgecolor('#2C3E50')
                    patch.set_facecolor(baseline_colors[label])
        
        for i, (arr, label, p_val) in enumerate(zip(ordered_data, labels, ordered_p_values)):
            mean = np.mean(arr)
            std = np.std(arr)
            if label != 'FedDQN':
                pooled_std = np.sqrt((np.var(ordered_data[0]) + np.var(arr)) / 2)
                effect_size = (mean - np.mean(ordered_data[0])) / pooled_std if pooled_std > 0 else 0
                sig = '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                improvement = ((mean - np.mean(ordered_data[0])) / abs(mean)) * 100 if mean != 0 else 0
                annotation = f"{mean:.2f}±{std:.2f}\n{sig}\nd={effect_size:.2f}\n↑{improvement:.1f}%"
            else:
                annotation = f"{mean:.2f}±{std:.2f}"
            
            plt.text(i, mean + std + 0.01, annotation,
                    ha='center', va='bottom', fontsize=12, color='#2C3E50', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.95, edgecolor='#95A5A6', 
                            pad=3, boxstyle='round,pad=0.5'))
        
        plt.title('Communication Overhead (MB)\nFedDQN vs Baseline Algorithms', 
                 fontsize=22, pad=20, color='#2C3E50', fontweight='bold')
        plt.ylabel('Overhead (MB)', fontsize=18, labelpad=10, color='#2C3E50', fontweight='bold')
        plt.xticks(rotation=30, ha='right', fontsize=16, color='#2C3E50')
        plt.yticks(fontsize=16, color='#2C3E50')
        
        plt.figtext(0.02, 0.02, 
                   f'Statistical Summary:\n* p < 0.05, ** p < 0.01\nViolins: Distribution with Quartiles\nd: Cohen\'s effect size\n↑: Improvement over FedDQN\nMean ± Std Dev: {mean:.2f} ± {std:.2f}',
                   fontsize=12, color='#2C3E50', fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.95, edgecolor='#95A5A6', 
                           pad=3, boxstyle='round,pad=0.5'))
        
        plt.grid(True, linestyle='--', alpha=0.3, color='#95A5A6')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'comm_overhead_results.png'), 
                   bbox_inches='tight', dpi=400, facecolor='white')
        plt.savefig(os.path.join(self.results_dir, 'comm_overhead_results.pdf'), 
                   bbox_inches='tight', dpi=600, facecolor='white')
        plt.close()

        plt.figure(figsize=(12, 12))
        
        metrics = ['latency', 'fairness', 'throughput', 'comm_overhead']
        num_metrics = len(metrics)
        
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]
        
        normalized_data = {}
        for metric in metrics:
            data = all_data[metric]
            values = np.array(data['means'])
            if metric in ['latency', 'comm_overhead']:
                values = 1 / (values + 1e-10)
            min_val = np.min(values)
            max_val = np.max(values)
            normalized_data[metric] = (values - min_val) / (max_val - min_val + 1e-10)
        
        ax = plt.subplot(111, polar=True)
        for i, label in enumerate(data['labels']):
            values = [normalized_data[metric][i] for metric in metrics]
            values += values[:1]
            color = feddqn_color if label == 'FedDQN' else baseline_colors[label]
            line = ax.plot(angles, values, 'o-', linewidth=3 if label=='FedDQN' else 2,
                         color=color, label=label, markersize=10 if label=='FedDQN' else 8,
                         markerfacecolor='white')
            if label == 'FedDQN':
                ax.fill(angles, values, color=color, alpha=0.2, hatch='////')
            else:
                ax.fill(angles, values, color=color, alpha=0.1)
        
        plt.xticks(angles[:-1], ['Latency\n(Lower Better)', 'Fairness\n(Higher Better)',
                                'Throughput\n(Higher Better)', 'Comm. Overhead\n(Lower Better)'],
                  fontsize=14)
        plt.title('Overall Performance Comparison\nFedDQN vs Baseline Algorithms', fontsize=22, pad=20)
        
        plt.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=14,
                  framealpha=0.9, edgecolor='gray', bbox_to_anchor=(1.3, 1.1))
        
        plt.figtext(0.02, 0.02, f'Statistical Summary:\n* p < 0.05, ** p < 0.01\nAll metrics normalized to [0,1]\nHigher values indicate better performance',
                   fontsize=12, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=2))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'algorithm_radar.png'), bbox_inches='tight', dpi=400)
        plt.savefig(os.path.join(self.results_dir, 'algorithm_radar.pdf'), bbox_inches='tight', dpi=600)
        plt.close()

        stats_df = pd.DataFrame(stats_rows)
        stats_df.to_csv(os.path.join(self.results_dir, 'experiment_stats.csv'), index=False)
        print("\nAll statistics saved to experiment_stats.csv")

if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run_experiment() 