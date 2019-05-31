# T-REX-IRL
**Trajectory-ranked Reward EXtrapolation (T-REX)** - A Tensorflow implementation trained on OpenAI Gym environments.

From the paper [**Extrapolating Beyond Suboptimal Demonstrations via Inverse Reinforcement Learning from Observations**](https://arxiv.org/abs/1904.06387).

T-REX is able to learn a reward function from a set of ranked low-scoring demonstrations, from which a policy can then be obtained (via a reinforcement learning algorithm) which significantly outperforms the suboptimal demonstrations.

![](https://i.imgur.com/DVnFssA.png)

This implementation has been trained and tested on [OpenAI Gym Atari environments](https://gym.openai.com/envs/#atari), achieving scores much greater than any seen in the original demonstrations, and greater even than the results obtained in the paper itself.

For the reinforcement learning algorithm (for generating the initial demonstrations and training the final policy on the learned reward function), the [OpenAI Baselines](https://github.com/openai/baselines) implementation of Proximal Policy Optimisation (PPO) is used, modified slightly to allow a choice between learning from the true reward from the environment (default) or instead supplying a learned reward function (the trained T-REX network).

## Requirements
Note: Versions stated are the versions I used, however this will still likely work with other versions.

- Ubuntu 16.04
- python 3.5
- [OpenAI Gym](https://github.com/openai/gym) 0.10.8 (See link for installation instructions + dependencies)
- [tensorflow-gpu](https://www.tensorflow.org/) 1.5.0
- [numpy](http://www.numpy.org/) 1.15.2
- [opencv-python](http://opencv.org/) 3.4.0

+ any other prerequisites for running OpenAI Baselines code, listed [here](https://github.com/openai/baselines#prerequisites)

## Usage
Note: This example will show usage for the 'Breakout' environment, to use any other environment simply modify the `--env` parameter.

- The first step is to train the default OpenAI Baselines PPO algorithm in the environment, frequently saving checkpoints (every `--save_interval` training updates) to be able to generate varying quality of demonstrations from different stages of the training:
```
  $ python -m baselines.run --alg=ppo2 --env='BreakoutNoFrameskip-v4' --save_interval=50
```
This will save the checkpoints in a folder in the `/tmp`' directory based on the time and date (e.g. `/tmp/openai-2019-05-27-18-26-59-016163/checkpoints`). Note that once the episode reward starts exceeding the reward of the demonstrations used in the paper, this training can be manually stopped (as we will not use any demonstrations which have a reward greater than those used in the paper, to make the results comparable). 


- The next step is to then generate the demonstration samples from these checkpoints:
```
  $ python generate_samples.py --env='Breakout' --ckpt_dir='/tmp/openai-2019-05-27-18-26-59-016163/checkpoints`
```


- The T-REX reward network is then trained on these demonstration samples, by running:
```
  $ python train.py --env='Breakout' --ckpt_dir='./ckpts/Breakout`
```
Note that this time the `--ckpt_dir` is where the checkpoints for the T-REX network should be saved.


- We then train the OpenAI Baselines PPO algorithm, similar to before, however this time using the learned reward function (the T-REX network) to provide the reward rather than the true environment reward. The algorithm will load the latest checkpoint in the `--reward_ckpt_dir` and use this network to provide the reward for training. As in the paper, we run for 50 million frames:
```
  $ python -m baselines.run --alg=ppo2 --env='BreakoutNoFrameskip-v4' --num_timesteps=50e6 --save_interval=5000 --learned_reward=True --reward_ckpt_dir=‘./ckpts/Breakout’ 
```
As before, this will save the checkpoints in a folder in the `/tmp`' directory based on the time and date (e.g. `/tmp/openai-2019-05-29-22-48-24-125657/checkpoints`).


- Finally, we can test the policy by running it in the environment and, as in the paper, taking the best average performance over 3 random seeds with 30 trials per seed:
```
  $ python test_learned_policy.py --env='Breakout' --ckpt_dir='/tmp/openai-2019-05-29-22-48-24-125657/checkpoints`
```

## Results
**Results from the paper:** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Results from this implementation:**

IMAGE

## Differences
There are some minor differences between this implementation of T-REX and that used in the paper:

- This implementation subsamples trajectory pairs from the saved demonstrations live during training; the paper does this offline as a preprocessing step before training - subsampling 6,000 trajectory pairs from the saved demonstrations then training on these subsamples.

- This implementation uses a fixed trajectory length of 50 when subsampling from the demonstrations; the paper chooses a random trajectory length each time between 50 and 100.

- This implementation trains on a batch of trajectory pairs at each step (batch size = 16), where a batch is made up of the unrolled states of the 16 trajectory pairs; the paper trains on a single trajectory pair at each step (where the 'batch' is just the unrolled trajectory states of 1 trajectory pair). 

## License
MIT License

