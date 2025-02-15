import os
os.environ['train_log_name']='0'
import subprocess
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from config import Config as cfg
from src.model import Agent
from src.get_data import get_done_data
import tensorflow as tf


if not cfg.gpu:
    tf.config.set_visible_devices([], 'GPU')
    print("GPUs have been disabled")
else:
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    print(gpu_devices)
    try:
        for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)
    except Exception as e:
        print('Could not set memory growth GPU') 


def main(train_log_name='results/train_log.csv'):
    """
    Main routine that runs in parallel for each experiment instance.
    """
    if not os.path.exists('weights/'):
        os.makedirs('weights')
    if os.path.exists('weights/deviser.npy'):
        os.remove('weights/deviser.npy')
    if os.path.exists('weights/decider.npy'):
        os.remove('weights/decider.npy')

    _, _, XV, BA = get_done_data(limit=15_000)
    
    agent=Agent(input_len = XV.shape[1:])    
    
    max_reward = 0
    test_rewards = []
    rewards = []

    # init
    _, _, XV_init, BA_init = get_done_data(limit=10)
    _, _, _, _ = agent.train_iteration(XV_init, BA_init, offline=False, online_learning=False)

    start = time.time()
    _, decisions, _, _ = agent.train_iteration(XV, BA,
                offline=False, 
                online_learning=False)

    end = time.time()
    print(f"RML Time: {end-start}")
    print(decisions[-50:])

    start = time.time()
    _, decisions, _, _ = agent.train_iteration(XV, BA,
                offline=True, 
                online_learning=False)

    end = time.time()
    print(f"Offline Time: {end-start}")
    print(decisions[-50:])

if __name__ == "__main__":
    main()
