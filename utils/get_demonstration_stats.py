'''
## Get demonstration stats ##
# Find the max and mean reward values across the set of demonstrations
@author: Mark Sinton (msinto93@gmail.com) 
'''

from data_loader import DataGenerator
import numpy as np
import argparse

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--env", type=str, help="Environment to use ('Breakout', 'SpaceInvaders', 'BeamRider', 'Q*bert', 'Hero', 'Pong')")
    return args.parse_args()

def get_stats(data_dir):
    reward_vals = []
    
    files = DataGenerator.list_np_files(data_dir)
            
    for file in files:
        reward_vals.append(DataGenerator.extract_reward(file))
        
    max_reward = np.max(reward_vals)
    mean_reward = np.average(reward_vals)
    
    return max_reward, mean_reward

if __name__ == '__main__':
    args = get_args()
    samples_dir = './samples/%s/train_data' % args.env
    
    max_reward, mean_reward = get_stats(samples_dir)
    
    print(args.env)
    print('Max reward = %.1f' % max_reward)
    print('Mean reward = %.1f' % mean_reward)

    