import random
import torch
import csv
import math
import pandas as pd
import numpy as np
from collections import namedtuple
from itertools import count
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(in_features=9, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=9)

    def forward(self, t):
        t = torch.relu(self.fc1(t))
        t = torch.relu(self.fc2(t))
        t = torch.relu(self.out(t))
        t = torch.sigmoid(t)
        return t

Experience = namedtuple(
    'Experience',
    ('state', 'next_state', 'reward')
)


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
               math.exp(-1 * current_step * self.decay)

class Agent:
    def __init__(self, strategy):
        self.current_step = 0
        self.strategy = strategy


    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            #print(torch.tensor(random.randrange(9)).item())
            return torch.tensor(random.randrange(9))
        else:
            with torch.no_grad():
                print(policy_net(state).argmax().item())
                return policy_net(state).argmax()

class GameManager:
    def __init__(self):
        self.reward = 0.5
        self.bot1 = 1
        self.bot2 = -1
        self.player = 1
        self.winner = 0
        self.board = torch.zeros(9)
        self.player = self.bot1
        self.done = False
        self.move_again = 0
        self.reset()

    def get_state(self):
        return self.board

    def reset(self):
        self.done = False
        self.player = 1
        self.reward = 0.5
        for i in range(9):
            self.board[i] = 0

    def avail_pos(self):
        for i in range(9):
            if self.board[i] == 0:
                return True

        return False

    def step(self, action):
        self.make_move(action.item())
        if self.reward == 0:
            self.done = True
        elif self.reward == -1:
            self.done = True
        elif self.reward == 1:
            self.done = True

    def game_over(self):
        if self.board[0] == self.board[1] == self.board[2] != 0:
            return True
        elif self.board[3] == self.board[4] == self.board[5] != 0:
            return True
        elif self.board[6] == self.board[7] == self.board[8] != 0:
            return True
        if self.board[0] == self.board[3] == self.board[6] != 0:
            return True
        elif self.board[1] == self.board[4] == self.board[7] != 0:
            return True
        elif self.board[2] == self.board[5] == self.board[8] != 0:
            return True
        elif self.board[0] == self.board[4] == self.board[8] != 0:
            return True
        elif self.board[2] == self.board[4] == self.board[6] != 0:
            return True

        return False

    def switch_bot(self):
        if self.player == self.bot1:
            self.player = self.bot2
        else:
            self.player = self.bot1

    def empty_cell(self, pos):
        if self.board[pos] == 0:
            return True

        return False

    def make_move(self, pos):
        if self.avail_pos():
            if self.empty_cell(pos):
                self.board[pos] = self.player
                if self.game_over():
                    self.reward = 1
                    return
                self.switch_bot()
                self.ran_move()
            else:
                ran_idx = random.randrange(9)
                self.make_move(ran_idx)
        else:
            self.reward = 0
            return

    def ran_move(self):
        if self.avail_pos():
            ran_idx = random.randrange(9)
            if self.empty_cell(ran_idx):
                self.board[ran_idx] = self.player
                if self.game_over():
                    self.reward = -1
                    return
                self.switch_bot()
            else:
                self.ran_move()
        else:
            self.reward = 0
            return

        return


def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))

    t1 = torch.stack(batch.state)
    t2 = torch.stack(batch.next_state)
    t3 = torch.stack(batch.reward)

    return t1, t2, t3


class QValues:
    @staticmethod
    def get_current(policy_net, states):
        value = policy_net(states).max(dim=1)
        return value

    @staticmethod
    def get_next(target_net, next_states):
        value = target_net(next_states).max(dim=1)
        return value

def train(num_episodes = 5000, batch_size = 256, policy_net = DQN(), target_net = DQN()):
    win_count = 0
    draw_count = 0
    lose_count = 0

    #batch_size = 256
    gamma = 0.100
    eps_start = 1
    eps_end = 0.001
    eps_decay = 0.00000000001
    target_update = 10
    memory_size = 100000
    lr = 0.01
    #num_episodes = 2000

    man = GameManager()
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    agent = Agent(strategy)
    memory = ReplayMemory(memory_size)

    #policy_net = DQN()
    #target_net = DQN()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

    episode_durations = []
    for episode in range(num_episodes):
        man.reset()
        state = man.get_state()
        y = state.detach().clone()

        for timestep in count():
            action = agent.select_action(y, policy_net)
            man.step(action)
            next_state = man.get_state()
            y1 = next_state.detach().clone()
            memory.push(Experience(y, y1, torch.tensor(man.reward)))
            y = next_state.detach().clone()

            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                states, next_states, rewards = extract_tensors(experiences)
                #print(policy_net(states).max(dim=1))
                current_q_values, current_q_indices = QValues.get_current(policy_net, states)
                next_q_values, next_q_indices = QValues.get_next(target_net, next_states)
                target_q_values = (next_q_values * gamma) + rewards

                loss = F.mse_loss(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #print(loss)

            if man.done:
                if man.reward == 1:
                    win_count = win_count + 1
                elif man.reward == -1:
                    lose_count = lose_count + 1
                elif man.reward == 0:
                    draw_count = draw_count + 1

                episode_durations.append(timestep)
                #print("episode over")
                break

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print(f"win: {win_count}")
    print(f"lose: {lose_count}")
    print(f"draw: {draw_count}")
    print(f"move_again: {man.move_again}")
    print(episode_durations)

    #torch.save(policy_net.state_dict(), "")
    #torch.save(target_net.state_dict(), "")

#print(episode_durations)

class HumanManager:
    def __init__(self):
        self.reward = 0.5
        self.bot1 = 1
        self.bot2 = -1
        self.player = 1
        self.winner = 0
        self.board = torch.zeros(9)
        self.player = self.bot1
        self.done = False
        self.reset()

    def get_state(self):
        return self.board

    def reset(self):
        self.done = False
        self.player = 1
        self.reward = 0.5
        for i in range(9):
            self.board[i] = 0

    def avail_pos(self):
        for i in range(9):
            if self.board[i] == 0:
                return True

        return False

    def step(self, action, pos1):
        self.make_move(action.item(), pos1)
        if self.reward == 0 or self.reward == -1 or self.reward == 1:
            self.done = True

    def game_over(self):
        if self.board[0] == self.board[1] == self.board[2] != 0:
            return True
        elif self.board[3] == self.board[4] == self.board[5] != 0:
            return True
        elif self.board[6] == self.board[7] == self.board[8] != 0:
            return True
        if self.board[0] == self.board[3] == self.board[6] != 0:
            return True
        elif self.board[1] == self.board[4] == self.board[7] != 0:
            return True
        elif self.board[2] == self.board[5] == self.board[8] != 0:
            return True
        elif self.board[0] == self.board[4] == self.board[8] != 0:
            return True
        elif self.board[2] == self.board[4] == self.board[6] != 0:
            return True

        return False

    def switch_bot(self):
        if self.player == self.bot1:
            self.player = self.bot2
        else:
            self.player = self.bot1

    def empty_cell(self, pos):
        if self.board[pos] == 0:
            return True

        return False

    def make_move(self, pos, pos1):
        if self.avail_pos():
            if not self.empty_cell(pos):
                self.reward = -1
            self.board[pos] = self.player
            if self.game_over():
                self.reward = 1
                return
            self.switch_bot()
            self.ran_move(pos1)
        else:
            self.reward = 0
            return

    def ran_move(self, pos1):
        if self.avail_pos():
            #ran_idx = random.randrange(9)
            #if not self.empty_cell(ran_idx):
                #self.ran_move()
            self.board[pos1] = self.player
            if self.game_over():
                self.reward = -1
                return
            self.switch_bot()
            return
        else:
            self.reward = 0
            return

class Agent1:
    def select_action(self, state, policy_net):
        with torch.no_grad():
            #print(policy_net(state).argmax().item())
            return policy_net(state).argmax()

def train1(policy_net):
    man = HumanManager()
    agent = Agent1()

    while(1):
        print("over")
        man.reset()
        state = man.get_state()
        y = state.detach().clone()
        while(not man.game_over()):
            action = agent.select_action(y, policy_net)

            print(action)
            inp = int(input("Enter ur val: "))
            man.step(action, inp)
            next_state = man.get_state()
            y = next_state.detach().clone()

def human_play():
    """
    model1 = DQN()
    model1.load_state_dict(torch.load(""))
    model1.eval()

    model = DQN()
    model.load_state_dict(torch.load(""))
    """

    policy_net = DQN()
    target_net = DQN()

    train(50000, 256, policy_net, target_net)

    train1(policy_net)

human_play()
