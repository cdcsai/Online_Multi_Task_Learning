import numpy as np
from sporco.admm import bpdn
from utils import reinitialize_zero_columns, discounted_r
import gym
import parameters


class Agent(object):

    def __init__(self, env, data_path):

        self.parameters = getattr(parameters, env)
        self.env = gym.make(self.parameters["env_name"])
        self.data_path = data_path

    def stochastic_policy(self, state, theta, sigma=1):
        mu = np.dot(theta.T, state)
        action = sigma * np.random.randn() + mu
        return action

    def get_random_trajectories(self, num_episodes=20, T_max=150):
        for e in range(1, num_episodes + 1):
            # Generate an episode
            episode = []
            state = self.env.reset()
            for t in range(T_max):
                action = np.random.randn(self.env.action_space)
                next_state, reward, done, = self.env.step(state, action)
                episode.append((state, action, reward))
                if done:
                    break
                state = next_state


        return episode

    def enac_policy(self, d, random, num_trajectories=20, T_max=150, alpha_1=0.1, alpha_2=0.1):
        theta = np.random.randn(d, 1)
        weights = np.random.randn(d, 1)

        for e in range(1, num_trajectories + 1):

            # Generate a trajectory
            trajectory = []
            state = self.env.reset()
            for t in range(T_max):
                if random:
                    action = np.random.randn(self.env.action_space)
                else:
                    action = Agent.stochastic_policy(state, theta)

                next_state, reward, done, = self.env.step(state, action)
                trajectory.append((state, action, reward))
                if done:
                    break
                state = next_state
        return alpha, hess

    def critic_evaluation(self):

        pass

    def actor_update(self):
        pass

    def OMLT(self, k, d, s, alpha, hess, dims, lmbda):
        A = np.zeros((k * d, k * d))
        b = np.zeros((k * d, 1))
        L = np.zeros((d, k))
        T = []

        while True:
            if t not in T:
                (T, R) = self.enac_policy(random=True)
            else:
                (T, R) = self.enac_policy(random=False)
                A = A - np.kron(np.dot(s, s.T), hess)
                temp = np.kron(s.T, np.dot(alpha, hess))
                b = b - temp.reshape(-1, 1)

            L = reinitialize_zero_columns(L)

            opt = bpdn.BPDN.Options()

            s = bpdn.BPDN(L, alpha, lmbda, opt)
            s = s.solve()

            A = A + np.kron(np.dot(s, s.T), hess)
            temp = np.kron(s.T, np.dot(alpha, hess))
            b = b + temp.reshape(-1, 1)
            L = np.reshape((dims))

    @ staticmethod
    def loss(L, s, alpha, hess, mu):
        t1 = mu * np.linalg.norm(s, ord=1)
        t2 = np.dot(np.dot((alpha - np.dot(L, s)).T, hess), (alpha - np.dot(L, s)))
        return t1 + t2

def get_random_trajectory():
    pass

def get_trajectories(alpha):
    pass

if __name__ == "__main__":
    pass