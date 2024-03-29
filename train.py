import gym
import time
import numpy as np
import dqn
import os
import agent
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def parse_args():
    parser = argparse.ArgumentParser('Reinforcement Learning parser for DQN')

    parser.add_argument('--scenario', type=str, default='LunarLander-v2')
    parser.add_argument('--eval', action='store_false')

    parser.add_argument('--load-episode-saved', type=int, default=50)
    parser.add_argument('--saved-episode', type=int, default=50)
    parser.add_argument('--update-time', type=int, default=4)
    parser.add_argument('--replay-buffer-size', type=int, default=int(1e5))

    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-episode', type=int, default=10000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=5e-4)

    return parser.parse_args()

try:  
    os.mkdir('./saved')
except OSError:  
    print ("Creation of the directory failed")
else:  
    print ("Successfully created the directory")

def main(arglist):
    env = gym.make(arglist.scenario)
    writer = SummaryWriter(log_dir='./logs/')

    actor = agent.Actor(env.observation_space.shape[0], env.action_space.n, arglist.lr, arglist.tau).to(device)
    actor.eval()
    target_actor = agent.Actor(env.observation_space.shape[0], env.action_space.n, arglist.lr, arglist.tau).to(device)
    target_actor.eval()

    dqn_algo = dqn.DQN(actor, target_actor, arglist.gamma, arglist.batch_size, arglist.replay_buffer_size, arglist.eval, arglist.update_time)
    dqn_algo.load('./saved/actor_' + str(arglist.load_episode_saved))

    t_step = 0
    for episode in range(arglist.max_episode):
        obs = env.reset()
        done = False
        j = 0
        losses = 0
        total_reward = 0
        while not done:
            if not arglist.eval:
                env.render()
            
            action, value_action = dqn_algo.act(obs)

            obs2, reward, done, info = env.step(action)
            total_reward += reward

            if arglist.eval:
                losses += dqn_algo.train(t_step, value_action, [reward], obs, obs2, [done])
            obs = obs2
            j += 1
            t_step += 1

        dqn_algo.epislon_decay()
        
        if arglist.eval and episode % arglist.saved_episode == 0 and episode > 0:
            actor.save_model('./saved/actor_' + str(episode))

        print('reward: ', total_reward, 'episode:', episode)
        if arglist.eval:
            writer.add_scalar('Loss', losses / float(j), episode)
            writer.add_scalar('Reward', total_reward, episode)
            writer.add_scalar('Epsilon_decay', dqn_algo.epsilon, episode)
        
    env.close()

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)