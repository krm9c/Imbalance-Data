import gym
import torch
import random
import numpy as np

from rl_class.pg.model import PolicyNet
from rl_class.pg.flags import getFlags
from rl_class.pg.event import event
# from gym.envs.registration import registry, register, make, spec
# register(
#     id='myenv-v0',
#     entry_point='gym.envs.algorithmic:myenv',
#     tags={'wrapper_config.TimeLimit.max_episode_steps': 500},
#     reward_threshold=25.0,
# )

def main(flags):

    env = gym.make(flags.task)
    env._max_episode_steps = flags.max_steps
    policy_net = PolicyNet()

    event_ = event(env, policy_net, flags)
    event_.model_train()
    event_.plot_end()


if __name__ == '__main__':
    args = getFlags()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
