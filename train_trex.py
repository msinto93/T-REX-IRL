'''
## Train T-REX ##
# Train the T-REX network on the demonstration samples
@author: Mark Sinton (msinto93@gmail.com) 
'''

import os
import sys
import tensorflow as tf
import numpy as np
import argparse

from utils.data_loader import DataGenerator
from utils.network import RewardNet
from utils.preprocess_trajs_offline import preprocess_trajs_offline

def get_args():
    args = argparse.ArgumentParser()
    
    args.add_argument("--env", type=str, help="Environment to use ('Breakout', 'SpaceInvaders', 'BeamRider', 'Q*bert', 'Hero', 'Pong')")
    args.add_argument("--ckpt_dir", type=str, help="Location to save T-REX checkpoints")
    args.add_argument("--restore_from_step", type=int, default=None, help="Checkpointed step to load and resume training from (if None, train from scratch)")
    args.add_argument("--n_train_steps", type=int, default=30000)
    args.add_argument("--val_trajs", type=int, default=1000, help="Number of trajectory pairs to generate before training to use as validation set")
    args.add_argument("--val_interval", type=int, default=500, help="Run validation every val_interval training steps and save checkpoint")
    args.add_argument("--early_stopping_threshold", type=int, default=10, help="Stop training after this number of validations without improvement")
    args.add_argument("--trn_batch_size", type=int, default=16)
    args.add_argument("--val_batch_size", type=int, default=1)
    args.add_argument("--learn_rate", type=float, default=5e-5)
    args.add_argument("--n_workers", type=int, default=8, help="For data loading and preprocessing")
    args.add_argument("--traj_length", type=int, default=50, help="We sample a random snippet of length traj_length from each demonstration sample to train on")
    return args.parse_args()

def train(train_data_dir, val_data_dir, ckpt_dir, restore_from_step, n_train_steps, val_interval, early_stopping_threshold, 
          trn_batch_size, val_batch_size, learn_rate, n_workers, traj_length):  
    
    # Initialise data generators
    train_datagen = DataGenerator(train_data_dir, batch_size=trn_batch_size, traj_len=traj_length, n_workers=n_workers, preprocessing_offline=False)
    val_datagen = DataGenerator(val_data_dir, batch_size=val_batch_size, traj_len=traj_length, n_workers=n_workers, preprocessing_offline=True)
    
    iterator = tf.data.Iterator.from_structure(train_datagen.data.output_types)
     
    train_init_op = iterator.make_initializer(train_datagen.data)
    val_init_op = iterator.make_initializer(val_datagen.data)
    
    # Get inputs and set shape so graph knows shape at compile time (shape is [batch_size, traj_len, frame_height, frame_width, num_frames])
    low_reward_traj, high_reward_traj, low_reward_value, high_reward_value = iterator.get_next()
    low_reward_traj.set_shape((None, traj_length, 84, 84, 4))
    high_reward_traj.set_shape((None, traj_length, 84, 84, 4))
    
    # Create network and training step op
    net = RewardNet()    
    reward_out_high = net.forward_pass(high_reward_traj)
    reward_out_low = net.forward_pass(low_reward_traj, reuse=True)
    
    opt = tf.train.AdamOptimizer(learn_rate)
    net.create_train_step(reward_out_high, reward_out_low, trn_batch_size, opt)
    
    # Create session and configure GPU options
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    # Set up saver to save/load ckpts
    saver = tf.train.Saver(max_to_keep=100)
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    if restore_from_step != None:
        ckpt_path = os.path.join(ckpt_dir, 'Step_%05d.ckpt' % (restore_from_step))
        print('Restoring from step %d' % (restore_from_step))
        saver.restore(sess, ckpt_path)
        start_step = restore_from_step + 1  
    else:
        print('No checkpoint file found. Initialising...')
        sess.run(tf.global_variables_initializer())
        start_step = 1 
    
    n_val_steps = len(val_datagen) // val_batch_size
   
    # Start training
    sess.run(train_init_op)
    train_losses = []
    best_val_acc = 0.0
    vals_without_improvement = 0
    
    print('\nTraining... \n')
    
    for train_step in range(start_step, n_train_steps+1):
        train_loss, _ = sess.run([net.loss, net.train_step])
        train_losses.append(train_loss)
        ave_train_loss = sum(train_losses)/len(train_losses)
        
        sys.stdout.write('\x1b[2K\rTrain Step: {:d}/{:d} \t Average Loss = {:.4f}'.format(train_step, n_train_steps, ave_train_loss))
        sys.stdout.flush()  
        
        if train_step % val_interval == 0:
            # Do validation
            sess.run(val_init_op)
            correct_preds = 0
            sys.stdout.write('\n')
            sys.stdout.flush()
            
            for val_step in range(1, n_val_steps+1):
                reward_high, reward_low = sess.run([reward_out_high, reward_out_low])
                reward_high_cum = np.sum(reward_high, axis=1)
                reward_low_cum = np.sum(reward_low, axis=1)
                
                for val_sample in range(val_batch_size):
                    if reward_high_cum[val_sample] > reward_low_cum[val_sample]:
                        correct_preds += 1
                        
                sys.stdout.write('\x1b[2K\rValidation Step: {:d}/{:d}'.format(val_step, n_val_steps))
                sys.stdout.flush()   
                        
            val_accuracy = (correct_preds / float(n_val_steps*val_batch_size)) * 100
            
            sys.stdout.write('\nValidation Complete. \t Accuracy = {:.2f}%'.format(val_accuracy))
            sys.stdout.flush()   
            
            if val_accuracy > best_val_acc:
                # Save ckpt
                ckpt_path = os.path.join(ckpt_dir, 'Step_%05d.ckpt' % train_step)
                saver.save(sess, ckpt_path)
                best_val_acc = val_accuracy
                # Reset early stopping counter
                vals_without_improvement = 0
            else:
                vals_without_improvement += 1
                            
            # Reinitialise training datagen and loss logging
            sess.run(train_init_op)
            train_losses = []
            print('\n')
        
        if vals_without_improvement == early_stopping_threshold:
            # Stop training
            sys.stdout.write('Training stopped after {:d} train steps due to {:d} consecutive validations without improvement.\n'.format(train_step, vals_without_improvement))
            sys.stdout.flush() 
            break

        
if __name__ == '__main__':
    args = get_args()
    
    train_samples_dir = './samples/%s/train_data' % args.env
    val_samples_dir = './samples/%s/val_data' % args.env
    
    # Run preprocessing function to generate validation trajectory pairs from demonstrations offline before training (ensures we have a fixed validation set as opposed to the dynamically generated training trajectories)
    preprocess_trajs_offline(val_samples_dir, args.val_trajs, args.traj_length)
    
    # Train
    train(train_samples_dir, val_samples_dir, args.ckpt_dir, args.restore_from_step, args.n_train_steps, args.val_interval, args.early_stopping_threshold,
          args.trn_batch_size, args.val_batch_size, args.learn_rate, args.n_workers, args.traj_length)
    
    
    
    