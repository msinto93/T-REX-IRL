'''
## Preprocess trajs offline ##
# Generate a fixed set of trajectory pairs offline before training from the set of demonstrations
@author: Mark Sinton (msinto93@gmail.com) 
'''

import numpy as np
import os
import sys
from random import shuffle

from utils.data_loader import DataGenerator

def preprocess_trajs_offline(data_dir, num_traj_pairs, traj_length):
    preprocessed_data_dir = data_dir + '_preprocessed'
    
    if not os.path.isdir(preprocessed_data_dir):
        os.makedirs(preprocessed_data_dir)
    
    existing_samples = len(os.listdir(preprocessed_data_dir))    
    
    if existing_samples != 0:
        print('\nData directory already contains %d samples. Exiting preprocessing.\n' % existing_samples)
        
    else:
        demonstrations = DataGenerator.list_np_files(data_dir)
        
        shuffle(demonstrations)
        
        num_demonstrations = len(demonstrations)
        traj_pairs = 0
        i = 0
        
        print('\nGenerating %d trajectory pairs from demonstrations...\n' % num_traj_pairs)
        
        while traj_pairs < num_traj_pairs:
            # When we get to the end of the set of demonstrations, reset iterator and shuffle list
            if i >= num_demonstrations - 1:
                i = 0
                shuffle(demonstrations)
            
            filenames = [demonstrations[i], demonstrations[i+1]]
            # Sort filenames into [lower_reward, higher_reward] order
            filenames = sorted(filenames, key=lambda x: DataGenerator.extract_reward(x))
            
            low_reward_value = DataGenerator.extract_reward(filenames[0])
            high_reward_value = DataGenerator.extract_reward(filenames[1])
            
            # Only process demonstrations further if they do not have equal rewards
            if low_reward_value != high_reward_value:
                low_reward_demo = np.load(filenames[0])['states'].copy()
                high_reward_demo = np.load(filenames[1])['states'].copy()
                
                # Randomly sample start indices of snippets (start_index_high must be > start_index_low)
                start_index_high = np.random.randint(4, high_reward_demo.shape[0]-traj_length)
                start_index_low = np.random.randint(3, min(start_index_high, low_reward_demo.shape[0]-traj_length))
                
                # Extract snippets, stacking previous 3 frames alongside current frame
                low_reward_snippet = np.concatenate((np.expand_dims(low_reward_demo[start_index_low-3:start_index_low+traj_length-3], axis=3),
                                                     np.expand_dims(low_reward_demo[start_index_low-2:start_index_low+traj_length-2], axis=3),
                                                     np.expand_dims(low_reward_demo[start_index_low-1:start_index_low+traj_length-1], axis=3),
                                                     np.expand_dims(low_reward_demo[start_index_low:start_index_low+traj_length], axis=3)), axis=3)
                                                     
                high_reward_snippet = np.concatenate((np.expand_dims(high_reward_demo[start_index_high-3:start_index_high+traj_length-3], axis=3),
                                                     np.expand_dims(high_reward_demo[start_index_high-2:start_index_high+traj_length-2], axis=3),
                                                     np.expand_dims(high_reward_demo[start_index_high-1:start_index_high+traj_length-1], axis=3),
                                                     np.expand_dims(high_reward_demo[start_index_high:start_index_high+traj_length], axis=3)), axis=3)
                
                # Set top 10 rows to 0 to mask game score
                low_reward_snippet[:, 0:10, :, :] = 0
                high_reward_snippet[:, 0:10, :, :] = 0
                
                np.savez(os.path.join(preprocessed_data_dir, 'Sample_%05d' % traj_pairs), low_reward_traj=low_reward_snippet, high_reward_traj=high_reward_snippet,
                                      low_reward_value=low_reward_value, high_reward_value=high_reward_value)
                
                sys.stdout.write('\rTrajectory %d/%d \tHigh reward = %.1f \tLow reward = %.1f' % (traj_pairs+1, num_traj_pairs, high_reward_value, low_reward_value))
                sys.stdout.flush()
                
                traj_pairs += 1
                i += 2
                
            else:
                i += 2
        
    
