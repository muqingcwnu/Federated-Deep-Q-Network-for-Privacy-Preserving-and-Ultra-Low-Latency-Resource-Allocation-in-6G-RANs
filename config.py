# Hyperparameters (Table 1)
HYPERPARAMS = {
    "learning_rate": 0.001,
    "discount_factor": 0.95,
    "replay_buffer_size": 10000,
    "batch_size": 64,
    "exploration_rate": 1.0,
    "exploration_min": 0.01,
    "exploration_decay": 0.995,
    "target_update_freq": 10,
    "fedavg_interval": 50
}

# Network Settings
NUM_BS = 5                   # Base stations
NUM_UE_PER_BS = 10           # User Equipment per BS
MAX_QUEUE_LENGTH = 100       # Packet buffer size
TRAFFIC_MODEL = "poisson"    # Poisson arrivals
CHANNEL_MODEL = "rayleigh"   # Rayleigh fading
TIME_SLOTS = 10000           # Simulation duration