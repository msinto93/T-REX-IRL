'''
## Environment Parameters ##
# Define environment parameters - 'reward_threshold' is the maximum reward from the demonstrations data in the paper, we only save samples
# with a reward smaller than this to make the results comparable with the paper
@author: Mark Sinton (msinto93@gmail.com) 
'''

env_params = {
    'Breakout': {
        'full_name': 'BreakoutNoFrameskip-v4',
        'reward_threshold': 32
        },
    'SpaceInvaders': {
        'full_name': 'SpaceInvadersNoFrameskip-v4',
        'reward_threshold': 600
        },
    'BeamRider': {
        'full_name': 'BeamRiderNoFrameskip-v4',
        'reward_threshold': 1332
        },
    'Pong': {
        'full_name': 'PongNoFrameskip-v4',
        'reward_threshold': -6
        },
    'Hero': {
        'full_name': 'HeroNoFrameskip-v4',
        'reward_threshold': 13235
        },
    'Qbert': {
        'full_name': 'QbertNoFrameskip-v4',
        'reward_threshold': 800
        }
    }