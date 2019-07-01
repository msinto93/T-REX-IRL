'''
## Test Learned Policy ##
# Test the policy generated from the learned reward function by running an agent in the environment following the policy
@author: Mark Sinton (msinto93@gmail.com) 
'''

import tensorflow as tf
import numpy as np
import functools
import os
import argparse

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.policies import build_policy
from baselines.common.tf_util import get_session, load_variables

from utils.env_params import env_params

def get_args():
    args = argparse.ArgumentParser()
    
    args.add_argument("--env", type=str, help="Environment to use ('Breakout', 'SpaceInvaders', 'BeamRider', 'Q*bert', 'Hero', 'Pong')")
    args.add_argument("--ckpt_dir", type=str, help="Location of checkpoints from PPO training")
    args.add_argument("--ckpt_step", type=int, default=None, help="Checkpoint step to load and test (if None, load latest step)")
    args.add_argument("--seeds", type=int, default=[1234, 12345, 123456], help="Random seeds to set for each trial")
    args.add_argument("--eps_per_seed", type=int, default=30, help="Number of episodes to run for each seed")
    args.add_argument("--max_ep_length", type=int, default=50000, help="Terminate episode after this many steps")
    args.add_argument("--render", type=bool, default=False, help="Display the environment on the screen during testing")
    
    return args.parse_args()


def test_policy(env_name, ckpt_dir, ckpt_step, seeds, eps_per_seed, max_ep_length, render):
    
    class Model(object):
        def __init__(self, policy, sess):
            with tf.variable_scope('ppo2_model'):
                act_model = policy(nbatch=1, nsteps=1, sess=sess)
                
            self.act_model = act_model
            self.step = act_model.step
            self.value = act_model.value
    
            self.load = functools.partial(load_variables, sess=sess)
    
    # Make environment
    env = make_atari(env_params[env_name]['full_name'])
    # For testing, to get accurate episode rewards, we do not clip rewards or end episode on loss of life
    env = wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True)
    
    # Build policy (actor) network
    policy = build_policy(env, policy_network='cnn')
    sess = get_session()
    model = Model(policy, sess)
    
    if ckpt_step == None:
        # Load latest checkpoint
        ckpts = sorted(os.listdir(ckpt_dir))
        latest_ckpt = ckpts[-1]
        load_path = os.path.join(ckpt_dir, latest_ckpt)
    else:
        load_path = os.path.join(ckpt_dir, str(ckpt_step))
    model.load(load_path)
    
    seed_rewards = []
    num_seeds = len(seeds)
    
    # Results are the best average performance over the random seeds
    for seed_count, seed in enumerate(seeds, start=1):
        env.seed(seed)
        rewards = []
    
        for ep in range(eps_per_seed):
        
            obs = env.reset()
            obs = np.expand_dims(obs, axis=0)   # expand_dims first converts LazyFrames object to Numpy array, then adds batch dimension            
            
            ep_reward = 0
            ep_done = False
            step = 0
            
            while not ep_done:
                step +=1 
                
                if render:
                    env.render()
                
                action, _, _, _ = model.step(obs)
                obs, reward, done, _ = env.step(action)
                
                obs = np.expand_dims(obs, axis=0)
        
                ep_reward += reward
                
                # Episode can finish either by reaching terminal state or max episode steps
                if done or step == max_ep_length:
                    ep_done = True 
            
            rewards.append(ep_reward)
            print('Ep: %02d \tReward: %.1f' % (ep, ep_reward))
        
        ave_reward = sum(rewards)/len(rewards)  
        seed_rewards.append(ave_reward)  
        print('\nSeed %d/%d complete. \tAverage Reward = %.2f\n' % (seed_count, num_seeds, ave_reward))
        
    best_reward = max(seed_rewards)
    print('Testing complete. \t Reward = %.2f' % (best_reward))
        

if __name__ == '__main__':
    args = get_args()
    
    test_policy(args.env, args.ckpt_dir, args.ckpt_step, args.seeds, args.eps_per_seed, args.max_ep_length, args.render)
    
        