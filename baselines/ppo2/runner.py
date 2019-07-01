import numpy as np
import tensorflow as tf
from baselines.common.runners import AbstractEnvRunner
from utils.network import RewardNet
from baselines.common.tf_util import get_session

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam, learned_reward=False, reward_ckpt_dir=None):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        # Use true reward function (from env) or learned reward function
        self.learned_reward = learned_reward
        
        if self.learned_reward:
            # Use a learned reward function (the TREX reward network) instead of the environment reward
            self.states_ph = tf.placeholder(tf.float32, (None, 84, 84, 4))
            self.reward_net = RewardNet()
            self.reward_out = self.reward_net.forward_pass(self.states_ph, reshape=False, sigmoid_out=True)
             
            self.sess = get_session()
             
            saver = tf.train.Saver(var_list=self.reward_net.network_params)
            
            ckpt_path = tf.train.latest_checkpoint(reward_ckpt_dir)
                             
            print('Restoring from %s' % ckpt_path)
            saver.restore(self.sess, ckpt_path)
        

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            
            if self.learned_reward:
                # Preprocess state to feed into reward network - Mask score, convert to float and normalise
                state_in = np.copy(self.obs)
                state_in[:, 0:10, :, :] = 0
                state_in = state_in.astype(np.float32, copy=False)
                state_in /= 255.0
     
                rewards = self.sess.run(self.reward_out, {self.states_ph:state_in})           
                rewards = np.squeeze(rewards)
                
                # Get everything else from the environment except reward
                self.obs[:], _, self.dones, infos = self.env.step(actions)
            
            else:       
                # Take actions in env and look the results
                # Infos contains a ton of useful informations
                self.obs[:], rewards, self.dones, infos = self.env.step(actions)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


