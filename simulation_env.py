import numpy as np
from collections import deque
from config import NUM_BS, NUM_UE_PER_BS, MAX_QUEUE_LENGTH

class RANEnvironment:
    def __init__(self, num_bs=NUM_BS, num_ue_per_bs=NUM_UE_PER_BS):
        self.num_bs = num_bs
        self.num_ue_per_bs = num_ue_per_bs
        self.reset()
    
    def reset(self):
        self.queues = [
            [deque(maxlen=MAX_QUEUE_LENGTH) for _ in range(self.num_ue_per_bs)] 
            for _ in range(self.num_bs)
        ]
        
        self.channel_gains = np.abs(
            np.random.randn(self.num_bs, self.num_ue_per_bs) + 
            1j*np.random.randn(self.num_bs, self.num_ue_per_bs)
        ) ** 2
        
        self.arrival_rates = np.random.uniform(0.1, 1.0, (self.num_bs, self.num_ue_per_bs))
        
        self.latency_accumulated = np.zeros((self.num_bs, self.num_ue_per_bs))
        self.packets_served = np.zeros((self.num_bs, self.num_ue_per_bs))
        
    def step(self, actions):
        if isinstance(actions, (np.ndarray, list)):
            actions = [int(a) for a in np.array(actions).flatten()]
        elif isinstance(actions, int):
            actions = [actions]
        else:
            actions = [int(actions)]

        if len(actions) != self.num_bs:
            raise ValueError(f"actions must have length {self.num_bs}, but got {len(actions)}: {actions}")
        
        self.channel_gains = np.abs(
            np.random.randn(self.num_bs, self.num_ue_per_bs) + 
            1j*np.random.randn(self.num_bs, self.num_ue_per_bs)
        ) ** 2
        
        new_packets = np.random.poisson(self.arrival_rates)
        for bs in range(self.num_bs):
            for ue in range(self.num_ue_per_bs):
                for _ in range(int(new_packets[bs, ue])):
                    if len(self.queues[bs][ue]) < MAX_QUEUE_LENGTH:
                        self.queues[bs][ue].append(0)
        
        for bs, action in enumerate(actions):
            ue = int(action) % self.num_ue_per_bs
            if self.queues[bs][ue]:
                arrival_time = self.queues[bs][ue].popleft()
                latency = -arrival_time
                self.latency_accumulated[bs, ue] += latency
                self.packets_served[bs, ue] += 1
        
        for bs in range(self.num_bs):
            for ue in range(self.num_ue_per_bs):
                for i in range(len(self.queues[bs][ue])):
                    self.queues[bs][ue][i] -= 1
        
        return self.get_state()
    
    def get_state(self):
        queue_state = np.array([
            [len(q) for q in bs_queues] 
            for bs_queues in self.queues
        ])
        return np.stack([queue_state, self.channel_gains], axis=-1)
    
    def get_metrics(self):
        total_latency = np.sum(self.latency_accumulated)
        total_packets = np.sum(self.packets_served)
        avg_latency = -total_latency / total_packets if total_packets > 0 else 0
        
        throughputs = self.packets_served.flatten()
        jain_index = (np.sum(throughputs) ** 2) / \
                     (self.num_bs * self.num_ue_per_bs * np.sum(throughputs ** 2))
        
        throughput = np.sum(throughputs)
        
        return avg_latency, jain_index, throughput