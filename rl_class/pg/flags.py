from distutils.util import strtobool
from argparse import ArgumentParser
import matplotlib
import numpy as np
import torch


def getFlags():

    parser = ArgumentParser(description='Argument Parser')

    parser.add_argument("--scratch", default=False, type=lambda x: bool(strtobool(x)),
                        help="Train the model from scratch")
    parser.add_argument("--seed", default=3407, type=int, help="Set random seed")
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str, help="Set up device")
    parser.add_argument("--optimizer", default="RMSprop", type=str, help="Optimizer")
    parser.add_argument("--is_ipython", default='inline' in matplotlib.get_backend(),
                        type=bool, help="Optimizer")
    parser.add_argument("--max_steps", default=100, type=int, help="Environment max steps for each episode")
    parser.add_argument("--loss", default='mse', type=str, help="Loss function")

    parser.add_argument("--num_episodes", default=10000, type=int, help="Number of episodes")
    parser.add_argument("--batch_size", default=5, type=int, help="Batch size")
    parser.add_argument("--learn_rate", default=1e-2, type=float, help="Learning rate")
    parser.add_argument("--gamma", default=0.99, type=float, help="Gamma")

    parser.add_argument("--save_after", default=100, type=int, help="Save model after number of episodes")
    parser.add_argument("--model_dir", default="./rl_class/pg/logs/", type=str,
                        help="Directory for saving model config files")
    parser.add_argument("--dataset_dir", default="./Data/dam/", type=str,
                        help="Directory for loading dataset")

    parser.add_argument("--task", default="CartPole-v0", type=str, help="Task to perform")
    parser.add_argument("--render", default=False, type=str,
                        help="Renders the environments to help visualise the agent actions")
    args = parser.parse_args()

    return args
