'''DLP DQN Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time
from datetime import timedelta

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from atari_wrappers import wrap_deepmind, make_atari



class ReplayMemory(object):
    ## TODO ##
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0


    def push(self, state, action, reward, next_state, done):
        """Saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = [state, action, reward, next_state, done]
        # print(len(self.memory[-1]))
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size, device):
        """Sample a batch of transitions"""
        transitions = random.sample(self.memory, batch_size)
        return (torch.tensor(np.array(x), dtype=torch.float, device=device)
                for x in zip(*transitions))

    def __len__(self):
        # return self.size
        return len(self.memory)


class Net(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(Net, self).__init__()

        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                 nn.ReLU(True)
                                        )
        self.classifier = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, num_classes)
                                        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # print(x.shape)
        x = x.permute(0, 3, 1, 2)
        x = x / 255.
        # print(x[:3,:3,:3])
        # print(x.shape)
        x = self.cnn(x)
        # print("after cnn")
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)


class DQN:
    def __init__(self, args):
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)
        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        self._target_net.eval()
        self._optimizer = torch.optim.Adam(self._behavior_net.parameters(), lr=args.lr, eps=1e-6)

        ## TODO ##
        """Initialize replay buffer"""
        self._memory = ReplayMemory(args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq

    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''
        ## TODO ##
        if random.random() < epsilon:
            return action_space.sample()
        
        with torch.no_grad():
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            # print(state.shape)
            outputs = self._behavior_net(state)
            best_action = torch.argmax(outputs)
            return best_action.item()

        

    def append(self, state, action, reward, next_state, done):
        ## TODO ##
        """Push a transition into replay buffer"""
        #self._memory.push(...)
        self._memory.push(state, [action], [reward], next_state, [int(done)])

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(self.batch_size, self.device)
        # print(state.shape, next_state.shape)
        ## TODO ##
        # new_list = [sublist[:-1] for sublist in original_list]
        q_value = self._behavior_net(state).gather(1, action.long())
        # print(len(q_value))
        with torch.no_grad():
            q_next = torch.max(self._target_net(next_state), 1)[0].view(-1, 1)
            # print(len(self._target_net(next_state)))
            q_target = reward + q_next * gamma * (1.0 - done)
        # print(torch.mean(q_target), torch.mean(reward), torch.mean(q_value))
        criterion = nn.MSELoss()
        loss = criterion(q_value, q_target)
        # optimize
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 2)
        self._optimizer.step()
        

    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        ## TODO ##
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
                'target_net': self._target_net.state_dict(),
                'optimizer': self._optimizer.state_dict(),
            }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path, self.device)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])

def train(args, agent, writer):
    print('Start Training')
    env_raw = make_atari('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env_raw, episode_life=True, clip_rewards=True, frame_stack=True)
    action_space = env.action_space
    total_steps, epsilon = 0, 0.1
    ewma_reward = 0
    max_reward  = 60

    for episode in range(25000, args.episode):
        total_reward = 0
        state = env.reset()
        # print(np.array(state).shape)
        state, reward, done, _ = env.step(1) # fire first !!!
        # state = torch.tensor(state).permute(2, 0, 1)
       
        for t in itertools.count(start=1):
            if total_steps < args.warmup:
                action = action_space.sample()
            else:
                # select action
                action = agent.select_action(np.array(state), epsilon, action_space)
                # decay epsilon
                epsilon -= (0.1 - args.eps_min) / args.eps_decay
                epsilon = max(epsilon, args.eps_min)

            # execute action
            state_, reward, done, _ = env.step(action)
            
            ## TODO ##
            # store transition
            #agent.append(...)
            agent.append(np.array(state), action, reward, np.array(state_), done)
            state = state_
            if total_steps >= args.warmup:
                agent.update(total_steps)

            total_reward += reward

            if (total_steps+1) % args.eval_freq == 0:
                """You can write another evaluate function, or just call the test function."""
                tmp = test(args, agent, writer)
                if tmp > max_reward:
                    max_reward = tmp
                    agent.save(args.model + "dqn-break-ex" + ".pt", True)

            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward, episode)
                writer.add_scalar('Train/Ewma Reward', ewma_reward, episode)
                print('Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                        .format(total_steps, episode, t, total_reward, ewma_reward, epsilon))
                break
    env.close()


def test(args, agent, writer):
    print('Start Testing')
    env_raw = make_atari('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env_raw, frame_stack=True)
    action_space = env.action_space
    e_rewards = []
    
    seed = args.seed if args.seed else random.randint(1, 100) 
    # print(f"Seed: {seed}")
    for i in range(args.test_episode):
        # states = deque(maxlen=4)
        env.seed(seed+i)
        state = env.reset()
        e_reward = 0
        done = False

        while not done:
            time.sleep(0.01)
            if args.render:
                env.render()
            action = agent.select_action(np.array(state), args.test_epsilon, action_space)
            # action = agent.select_action(states, args.test_epsilon, action_space)
            state, reward, done, _ = env.step(action)
            e_reward += reward
        print('episode {}: {:.2f}'.format(i+1, e_reward))
        e_rewards.append(e_reward)

    env.close()
    print('Average Reward: {:.2f}'.format(float(sum(e_rewards)) / float(args.test_episode)))
    print(f"Got {5 * ((float(sum(e_rewards)) / float(args.test_episode)) ** 0.5)} in lab!")
    return 5 * ((float(sum(e_rewards)) / float(args.test_episode)) ** 0.5)


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='ckpt/')
    parser.add_argument('--logdir', default='log/dqn_breakout')
    # train
    parser.add_argument('--warmup', default=20000, type=int)
    parser.add_argument('--episode', default=200000, type=int)
    parser.add_argument('--capacity', default=200000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.0000625, type=float)
    parser.add_argument('--eps_decay', default=500000, type=float)
    parser.add_argument('--eps_min', default=0.1, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=1000, type=int)
    parser.add_argument('--eval_freq', default=200000, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('-tmp', '--test_model_path', default='dqn-break-ex.pt')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--test_episode', default=10, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--test_epsilon', default=0.01, type=float)
    args = parser.parse_args()

    ## main ##
    agent = DQN(args)
    if args.test_only:
        writer = SummaryWriter("log/dqn_breakout_test")
    else:
        writer = SummaryWriter(args.logdir)
    if args.test_only:
        agent.load(args.test_model_path)
        test(args, agent, writer)
        # highest = []
        # for _ in range(10):
        #     highest += [test(args, agent, writer)]
        # print(f"Highest {np.max(highest)} at {np.argmax(highest)} round")
    else:
        # agent.load(args.test_model_path, True)
        train(args, agent, writer)
        


if __name__ == '__main__':
    start = time.time()
    main()
    print(f"Executed in {timedelta(seconds=time.time() - start)}")
