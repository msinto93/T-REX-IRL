'''
## Generate Demonstrations ##
# Generate the demonstration samples by running the trained agents in the environment and saving the frames
@author: Mark Sinton (msinto93@gmail.com) 
'''

import tensorflow as tf
import numpy as np
import functools
import os
import cv2
import argparse

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.policies import build_policy
from baselines.common.tf_util import get_session, load_variables

from utils.env_params import env_params

def get_args():
    args = argparse.ArgumentParser()
    
    args.add_argument("--env", type=str, help="Environment to use ('Breakout', 'SpaceInvaders', 'BeamRider', 'Q*bert', 'Hero', 'Pong')")
    args.add_argument("--ckpt_dir", type=str, help="Location of checkpoints from PPO training")
    args.add_argument("--num_eps", type=int, default=12, help="Number of episodes to run each checkpointed model for")
    args.add_argument("--max_ep_length", type=int, default=50000, help="Terminate episode after this many steps")
    args.add_argument("--visualise_after", type=bool, default=True, help="Visualise the samples after generating them to sanity check they look correct")
    
    return args.parse_args()

    
def generate_samples(env_name, train_samples_dir, val_samples_dir, ckpt_dir, num_eps, max_ep_length):

    class Model(object):
        def __init__(self, policy, sess):
            with tf.variable_scope('ppo2_model'):
                act_model = policy(nbatch=1, nsteps=1, sess=sess)
                
            self.step = act_model.step
            self.load = functools.partial(load_variables, sess=sess)
    
    if not os.path.isdir(train_samples_dir):
        os.makedirs(train_samples_dir)
    if not os.path.isdir(val_samples_dir):
        os.makedirs(val_samples_dir)
    
    # Make environment
    env = make_atari(env_params[env_name]['full_name'])
    # To get accurate episode rewards, we do not clip rewards or end episode on loss of life (only do these during training)
    env = wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True)
    
    # Build policy (actor) network
    policy = build_policy(env, policy_network='cnn')
    sess = get_session()
    model = Model(policy, sess)
    
    ckpts = sorted(os.listdir(ckpt_dir))
        
    for train_step in ckpts:
        
        load_path = os.path.join(ckpt_dir, train_step)
        model.load(load_path)
        
        # Choose one episode at random for each ckpt to use as the validation sample
        val_ep = np.random.randint(0, num_eps)
        
        for ep in range(num_eps):
        
            obs = env.reset()
            obs = np.expand_dims(obs, axis=0)   # expand_dims first converts LazyFrames object to Numpy array, then adds batch dimension
            # Create states array to store all the obs arrays
            states = np.copy(obs[:, :, :, -1])       # Only store the current frame not the previous 3, to avoid having to store every frame 4 times        
            
            ep_reward = 0
            ep_done = False
            step = 0
            
            while not ep_done:
                step +=1 
                
                action, _, _, _ = model.step(obs)
                obs, reward, done, _ = env.step(action)
                
                obs = np.expand_dims(obs, axis=0)
                states = np.concatenate((states, obs[:, :, :, -1]), axis=0)
        
                ep_reward += reward
                
                # Episode can finish either by reaching terminal state or max episode steps
                if done or step == max_ep_length:
                    ep_done = True 
            
            if ep_reward > env_params[env_name]['reward_threshold']:
                # Don't save samples with a reward greater than the maximum reward of the demonstrations used in the paper, to allow for comparable results
                continue
            
            print('Train Step: %s \tEp: %02d \tReward: %.1f' % (train_step, ep, ep_reward))
            
            if ep == val_ep:
                np.savez(os.path.join(val_samples_dir, 'Step_%s_Ep_%02d_Reward_%.1f' % (train_step, ep, ep_reward)), states=states)
            else:
                np.savez(os.path.join(train_samples_dir, 'Step_%s_Ep_%02d_Reward_%.1f' % (train_step, ep, ep_reward)), states=states)
                

def visualise_samples(samples_dir):
    for sample in os.listdir(samples_dir):
        print(sample)
        sample = os.path.join(samples_dir, sample)
        states = np.load(sample)['states']
        
        for state_num in range(states.shape[0]):
            frame = states[state_num]
            cv2.imshow('Frame', frame) 
            cv2.waitKey(10)
        

if __name__ == '__main__':
    args = get_args()
    
    train_samples_dir = './samples/%s/train_data' % args.env
    val_samples_dir = './samples/%s/val_data' % args.env
        
    generate_samples(args.env, train_samples_dir, val_samples_dir, args.ckpt_dir, args.num_eps, args.max_ep_length)
    
    if args.visualise_after:
        visualise_samples(train_samples_dir)
        visualise_samples(val_samples_dir)
        