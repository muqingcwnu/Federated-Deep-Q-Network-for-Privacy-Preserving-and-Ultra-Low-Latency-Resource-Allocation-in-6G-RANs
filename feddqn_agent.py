import tensorflow as tf
from tensorflow.keras import layers, optimizers
import numpy as np
import random
from collections import deque
from config import *

class DQNAgent:
    def __init__(self, state_shape, action_size, agent_id):
        self.state_shape = state_shape
        self.action_size = action_size
        self.id = agent_id
        
        self.gamma = HYPERPARAMS["discount_factor"]
        self.optimizer = optimizers.Adam(HYPERPARAMS["learning_rate"])
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.model.compile(optimizer=self.optimizer, loss='mse')
        self.target_model.compile(optimizer=self.optimizer, loss='mse')
        
        self.replay_buffer = deque(maxlen=HYPERPARAMS["replay_buffer_size"])
        self.batch_size = HYPERPARAMS["batch_size"]
        
        self.epsilon = HYPERPARAMS["exploration_rate"]
        self.epsilon_min = HYPERPARAMS["exploration_min"]
        self.epsilon_decay = HYPERPARAMS["exploration_decay"]
        
        self.target_update_counter = 0
        
    def _build_model(self):
        inputs = layers.Input(shape=self.state_shape)
        x = layers.Flatten()(inputs)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(self.action_size, activation='linear')(x)
        return tf.keras.Model(inputs, outputs)
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(np.expand_dims(state, axis=0))
        return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = np.array(states)
        next_states = np.array(next_states)
        
        current_q = self.model.predict(states)
        next_q = self.target_model.predict(next_states)
        max_next_q = np.amax(next_q, axis=1)
        
        for i in range(self.batch_size):
            if dones[i]:
                current_q[i][actions[i]] = rewards[i]
            else:
                current_q[i][actions[i]] = rewards[i] + self.gamma * max_next_q[i]
        
        self.model.train_on_batch(states, current_q)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.target_update_counter += 1
        if self.target_update_counter >= HYPERPARAMS["target_update_freq"]:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

class CentralizedDQN(DQNAgent):
    def __init__(self, state_shape, action_size, num_bs):
        self.num_bs = num_bs
        super().__init__(state_shape, action_size, agent_id=-1)
        
    def _build_model(self):
        inputs = layers.Input(shape=self.state_shape)
        x = layers.Flatten()(inputs)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        outputs = layers.Dense(self.num_bs * self.action_size, activation='linear')(x)
        return tf.keras.Model(inputs, outputs)
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return [np.random.randint(self.action_size) for _ in range(self.num_bs)]
            
        q_values = self.model.predict(np.expand_dims(state, axis=0))[0]
        q_matrix = q_values.reshape((self.num_bs, self.action_size))
        return np.argmax(q_matrix, axis=1)

class FedAvgAggregator:
    def __init__(self, agents):
        self.agents = agents
    
    def aggregate(self):
        global_weights = []
        num_samples = [len(agent.replay_buffer) for agent in self.agents]
        total_samples = sum(num_samples)
        
        for layer_idx in range(len(self.agents[0].model.get_weights())):
            layer_weights = []
            for agent, n_i in zip(self.agents, num_samples):
                layer_weights.append(agent.model.get_weights()[layer_idx] * n_i)
            global_layer = np.sum(layer_weights, axis=0) / total_samples
            global_weights.append(global_layer)
        
        for agent in self.agents:
            agent.model.set_weights(global_weights)
            agent.target_model.set_weights(global_weights)

class FedDQNAgent:
    def __init__(self, is_centralized=False):
        self.is_centralized = is_centralized
        if is_centralized:
            self.agent = CentralizedDQN(
                state_shape=(NUM_BS, NUM_UE_PER_BS, 2),
                action_size=NUM_UE_PER_BS,
                num_bs=NUM_BS
            )
        else:
            self.agents = [
                DQNAgent(state_shape=(NUM_UE_PER_BS, 2), action_size=NUM_UE_PER_BS, agent_id=bs)
                for bs in range(NUM_BS)
            ]
            self.aggregator = FedAvgAggregator(self.agents)
        self.comm_overhead = 0
        self.step_counter = 0

    def act(self, state):
        if self.is_centralized:
            return self.agent.act(state)
        else:
            actions = []
            for bs, agent in enumerate(self.agents):
                bs_state = state[bs]
                actions.append(agent.act(bs_state))
            return actions

    def train(self, state, action, reward, next_state):
        if self.is_centralized:
            self.agent.remember(state, action, reward, next_state, done=False)
            self.agent.replay()
        else:
            for bs, agent in enumerate(self.agents):
                bs_state = state[bs]
                bs_action = action[bs]
                bs_reward = reward
                bs_next_state = next_state[bs]
                agent.remember(bs_state, bs_action, bs_reward, bs_next_state, done=False)
                agent.replay()
            self.step_counter += 1
            if self.step_counter % HYPERPARAMS["fedavg_interval"] == 0:
                self.aggregator.aggregate()
                weights_size = sum([w.size for w in self.agents[0].model.get_weights()])
                self.comm_overhead += weights_size * NUM_BS * 4 / 1e6

    def get_comm_overhead(self):
        return self.comm_overhead