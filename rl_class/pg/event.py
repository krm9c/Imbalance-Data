import os
import torch
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import get_time
from itertools import count
import torch.optim as optimize
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.distributions import Bernoulli

class event:
    def __init__(self, env, policy_net, args):

        self.n_actions = env.action_space.n
        self.policy_net = policy_net
        self.env = env

        self.episode_durations = []
        self.state_pool = []
        self.action_pool = []
        self.reward_pool = []
        self.steps = 0

        self.num_episodes = args.num_episodes
        self.is_ipython = args.is_ipython
        self.save_after = args.save_after
        self.batch_size = args.batch_size
        self.model_dir = args.model_dir
        self.scratch = args.scratch
        self.render = args.render
        self.gamma = args.gamma
        self.seed = args.seed

        if args.optimizer == 'RMSprop':
            self.optimizer = optimize.RMSprop(policy_net.parameters())
        elif args.optimizer == 'Adam':
            self.optimizer = optimize.Adam(policy_net.parameters(), lr=args.learn_rate)
        elif args.optimizer == 'sgd':
            self.optimizer = optimize.sgd(policy_net.parameters())
        else:
            raise 'Unable to recognize the optimizer'
        pass

    def model_train(self):

        if not self.scratch:
            self.model_load()

        pbar = tqdm(range(self.num_episodes), dynamic_ncols=True, smoothing=0.1)

        for e in pbar:

            state = self.env.reset()
            state = torch.from_numpy(state).float()
            state = Variable(state)
            if self.render:
                self.env.render(mode="rgb_array")

            # Sample the trajectory
            self.sample_trajectory(state)
            pbar.set_postfix({
                'Reward': '{0:1.1f}'.format(sum(self.reward_pool))})

            # Update policy
            self.policy_update()

            if (e + 1) % self.save_after == 0:
                self.model_save()
                df = pd.DataFrame(self.episode_durations)
                df.to_csv(os.path.join(self.model_dir, 'durations.csv'))

    def sample_trajectory(self, state):

        for t in count():

            probs = self.policy_net(state)
            m = Bernoulli(probs)
            action = m.sample()

            action = action.data.numpy().astype(int)[0]
            next_state, reward, done, _ = self.env.step(action)
            if self.render:
                self.env.render(mode="rgb_array")

            # To mark boundary between episodes
            if done:
                reward = 0

            self.state_pool.append(state)
            self.action_pool.append(float(action))
            self.reward_pool.append(reward)

            state = next_state
            state = torch.from_numpy(state).float()
            state = Variable(state)

            self.steps += 1


            if done:
                self.episode_durations.append(t + 1)
                break

    def policy_update(self):

        # Discount reward
        running_add = 0
        for i in reversed(range(self.steps)):
            if self.reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.gamma + self.reward_pool[i]
                self.reward_pool[i] = running_add

        # Normalize reward
        reward_mean = np.mean(self.reward_pool)
        reward_std = np.std(self.reward_pool)
        for i in range(self.steps):
            self.reward_pool[i] = (self.reward_pool[i] - reward_mean) / reward_std

        # Policy update
        self.optimizer.zero_grad()

        for i in range(self.steps):
            state = self.state_pool[i]
            action = Variable(torch.FloatTensor([self.action_pool[i]]))
            reward = self.reward_pool[i]

            probs = self.policy_net(state)
            m = Bernoulli(probs)
            loss = -m.log_prob(action) * reward  # Negative score function x reward
            loss.backward()

        self.optimizer.step()
        self.reset_pool()

    # def plot_durations(self):
    #     plt.ion()
    #     plt.figure(1)
    #     plt.clf()
    #     durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
    #     plt.title('Training...')
    #     plt.xlabel('Episode')
    #     plt.ylabel('Duration')
    #     plt.plot(durations_t.numpy())
    #     # Take 100 episode averages and plot them too
    #     if len(durations_t) >= 100:
    #         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #         means = torch.cat((torch.zeros(99), means))
    #         plt.plot(means.numpy())
    #
    #     plt.pause(0.001)  # pause a bit so that plots are updated
    #     # if self.is_ipython:
    #     #     display.clear_output(wait=True)
    #     #     display.display(plt.gcf())

    def plot_end(self):
        plt.figure(figsize=(8, 6), dpi=150)
        plt.plot(self.episode_durations, linewidth=5)
        plt.xlabel('Episode')
        plt.ylabel('# Steps')
        plt.savefig(os.path.join(self.model_dir, 'fig', 'fig' + str(self.seed) + get_time() + '.png'))

    def model_load(self):

        try:
            model_list = os.listdir(self.model_dir + '/net/')
            model_list.sort(key=lambda fn: os.path.getmtime(self.model_dir + '/net/' + fn))
            datetime.datetime.fromtimestamp(os.path.getmtime(self.model_dir + '/net/' + model_list[-1]))
            filepath = os.path.join(self.model_dir, 'net', model_list[-1])
            self.policy_net.load_state_dict(torch.load(filepath))
            print('Load previous model: ' + filepath + 'successfully!')
        except IndexError:
            print('No trained network. Training network from scratch')
        except FileNotFoundError:
            print('No trained network. Training network from scratch')
            self.policy_net.initialize()

    def model_save(self):

        try:
            torch.save(self.policy_net.state_dict(), self.get_policy_path())
        except FileNotFoundError:
            os.makedirs(self.model_dir)
            torch.save(self.policy_net.state_dict(), self.get_policy_path())
        print("Save model: " + self.get_policy_path() + " successfully.")

    def get_policy_path(self):
        return os.path.join(self.model_dir, 'net', 'policy_model_' + get_time() + '.pth')

    def reset_pool(self):
        self.state_pool = []
        self.action_pool = []
        self.reward_pool = []
        self.steps = 0
