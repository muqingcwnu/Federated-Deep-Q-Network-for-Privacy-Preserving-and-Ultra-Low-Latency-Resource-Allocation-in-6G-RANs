import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from config import NUM_BS, NUM_UE_PER_BS, HYPERPARAMS

class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = layers.Dense(400, activation='relu')
        self.fc2 = layers.Dense(300, activation='relu')
        self.fc3 = layers.Dense(action_dim, activation='tanh')
        
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.fc3(x)

class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = layers.Dense(400, activation='relu')
        self.fc2 = layers.Dense(300, activation='relu')
        self.fc3 = layers.Dense(1)
        
    def call(self, state, action):
        x = tf.concat([state, action], axis=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=HYPERPARAMS['learning_rate'])
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=HYPERPARAMS['learning_rate'])
        
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
    def get_action(self, state):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action = self.actor(state)
        return action.numpy()

class PFS:
    def __init__(self):
        pass
        
    def schedule(self, channel_gains, queue_lengths):
        scores = channel_gains / (queue_lengths + 1e-6)
        return np.argmax(scores, axis=1)

def pfs_scheduler(channel_gains, queue_lengths):
    scores = channel_gains / (queue_lengths + 1e-6)
    return np.argmax(scores, axis=1)

class RoundRobinScheduler:
    def __init__(self, num_ues):
        self.current_ue = [0] * NUM_BS
        self.num_ues = num_ues
    
    def schedule(self):
        actions = []
        for bs in range(NUM_BS):
            actions.append(self.current_ue[bs])
            self.current_ue[bs] = (self.current_ue[bs] + 1) % self.num_ues
        return actions

def static_average_scheduler():
    return [np.random.randint(0, NUM_UE_PER_BS) for _ in range(NUM_BS)]

class DDPGScheduler:
    def __init__(self, state_dim, action_dim):
        self.agent = DDPGAgent(state_dim, action_dim)
        
    def schedule(self, state):
        actions = []
        for bs in range(NUM_BS):
            bs_state = state[bs]
            state_input = np.expand_dims(bs_state.flatten(), axis=0)
            action = self.agent.get_action(state_input)
            if isinstance(action, np.ndarray):
                action = int(np.clip(np.round(action[0][0]), 0, NUM_UE_PER_BS-1))
            actions.append(action)
        return actions