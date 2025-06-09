import numpy as np
from feddqn_agent import DQNAgent, FedAvgAggregator, CentralizedDQN
from baselines import pfs_scheduler, RoundRobinScheduler, static_average_scheduler
from simulation_env import RANEnvironment
from config import *
import sys

def simple_progress(iterable, desc="", total=None):
    """Custom progress bar replacement"""
    total = total or len(iterable)
    for i, item in enumerate(iterable):
        if i % 100 == 0:
            sys.stdout.write(f"\r{desc} {i}/{total} ({i/total:.0%})")
            sys.stdout.flush()
        yield item
    sys.stdout.write("\r" + " " * 50 + "\r")
    sys.stdout.flush()

# Initialize environment and agents
env = RANEnvironment(NUM_BS, NUM_UE_PER_BS)
state_shape = (NUM_UE_PER_BS, 2)  # Per-BS state shape (queue_len, channel_gain)

# Create agents
fed_agents = [DQNAgent(state_shape, NUM_UE_PER_BS, i) for i in range(NUM_BS)]
aggregator = FedAvgAggregator(fed_agents)
rr_scheduler = RoundRobinScheduler(NUM_UE_PER_BS)

# Create centralized DQN for comparison
global_state_shape = (NUM_BS, NUM_UE_PER_BS, 2)
c_dqn = CentralizedDQN(global_state_shape, NUM_UE_PER_BS, NUM_BS)

# Training loop
for time_slot in simple_progress(range(TIME_SLOTS), desc="Training"):
    # 1. Get current state
    state = env.get_state()
    
    # 2. FedDQN agents take actions
    actions = [agent.act(state[bs]) for bs, agent in enumerate(fed_agents)]
    
    # 3. Execute actions in environment
    next_state = env.step(actions)
    
    # 4. Compute reward
    latency = env.latency_accumulated.copy()
    throughput = env.packets_served.copy()
    resource_cost = np.sum([len(q) for bs in env.queues for q in bs])
    
    rewards = [
        0.7 * (-latency[bs, actions[bs]]) + 
        0.2 * throughput[bs, actions[bs]] - 
        0.1 * resource_cost
        for bs in range(NUM_BS)
    ]
    
    # 5. Store experience and train
    for bs in range(NUM_BS):
        fed_agents[bs].remember(
            state[bs], 
            actions[bs], 
            rewards[bs], 
            next_state[bs], 
            time_slot >= TIME_SLOTS - 1
        )
        fed_agents[bs].replay()
    
    # 6. Federated averaging
    if time_slot % HYPERPARAMS["fedavg_interval"] == 0:
        aggregator.aggregate()
    
    # 7. Periodic evaluation
    if time_slot % 1000 == 0 and time_slot > 0:
        # FedDQN metrics
        fed_latency, fed_fairness, fed_throughput = env.get_metrics()
        
        # Baseline comparisons
        env.step(pfs_scheduler(env.channel_gains, 
                             np.array([[len(q) for q in bs] for bs in env.queues])))
        pfs_latency, pfs_fairness, pfs_throughput = env.get_metrics()
        
        env.step(rr_scheduler.schedule())
        rr_latency, rr_fairness, rr_throughput = env.get_metrics()
        
        env.step(static_average_scheduler())
        sa_latency, sa_fairness, sa_throughput = env.get_metrics()
        
        # Centralized DQN
        c_actions = c_dqn.act(state)
        env.step(c_actions)
        c_latency, c_fairness, c_throughput = env.get_metrics()
        
        print(f"\nStep {time_slot}:")
        print(f"FedDQN: {fed_latency:.2f}ms | {fed_fairness:.3f} fairness | {fed_throughput:.1f} Mbps")
        print(f"C-DQN: {c_latency:.2f}ms | {c_fairness:.3f} fairness | {c_throughput:.1f} Mbps")
        print(f"PFS: {pfs_latency:.2f}ms | {pfs_fairness:.3f} fairness | {pfs_throughput:.1f} Mbps")
        print(f"RR: {rr_latency:.2f}ms | {rr_fairness:.3f} fairness | {rr_throughput:.1f} Mbps")
        print(f"SA: {sa_latency:.2f}ms | {sa_fairness:.3f} fairness | {sa_throughput:.1f} Mbps")

# Save models
for bs, agent in enumerate(fed_agents):
    agent.model.save(f"fedqdn_bs_{bs}.h5")
c_dqn.model.save("centralized_dqn.h5")