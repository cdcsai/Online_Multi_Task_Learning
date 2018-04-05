import numpy as np
from utils import reinitialize_zero_columns, discounted_r, loss, hess_norm
import gym
from gym_extensions.continuous import gym_navigation_2d
from scipy.optimize import minimize
from copy import deepcopy
import copy
from collections import defaultdict


class Agent(object):
    """
    Online Learner able to learn efficiently several tasks.
    """

    def __init__(self):

        self.env = gym.make("State-Based-Navigation-2d-Map1-Goal1-v0")
        self.render = False
        self.task1, self.task2, self.task3 = self.create_multitask_envs
        self.nb_tasks = 3
        self.discount = 0.9
        self.sigma = 0.8
        self.learning_rate = 0.0001
        self.obs_space_shape = self.env.observation_space.shape[0]
        self.action_space_shape = self.env.action_space.shape[0]
        self.theta = np.random.randn(self.obs_space_shape)
        self.nb_episodes = 10
        self.horizon = 5
        self.lbda = 0.1

    @property
    def create_multitask_envs(self):
        env1 = deepcopy(self.env)
        env2 = deepcopy(self.env)
        env3 = deepcopy(self.env)

        env1.env.destination = np.array([20, 100])
        env2.env.destination = np.array([300, 350])
        env3.env.destination = np.array([500, 500])
        if self.render:
            env1.render()
            env2.render()
            env3.render()
        return env1, env2, env3

    def collect_episodes(self, mdp, policy=None, horizon=None, n_episodes=1):

        """
        Takes a Markov Decision Process in input and returns the collected actions, rewards and observations of the
        simulation.

        Parameters:
        mdp -- Markov Decision Process (or environment)

        Returns:
        paths -- the collected path in a dictionnary form
        """

        paths = []

        for _ in range(n_episodes):
            observations = []
            actions = []
            rewards = []
            next_states = []

            state = mdp.reset()
            # state = state.reshape(-1, 1)

            for _ in range(horizon):
                action = policy.draw_action(state, mdp)
                next_state, reward, terminal, _ = mdp.step(action)
                if self.render:
                    mdp.render()
                observations.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                state = copy.copy(next_state)
                if terminal:
                    break

            paths.append(dict(
                states=np.array(observations),
                actions=np.array(actions),
                rewards=np.array(rewards),
                next_states=np.array(next_states)
            ))
        return paths

    def gaussian_grad(self, action, state, theta):
        """
        Compute the gradient of the gaussian policy. Used in the reinforce policy gradient method.

        Parameters:
        current action, current state, current theta

        Returns:
        The updated gradient
        """

        return np.squeeze(np.outer(state, (action - np.dot(theta.T, state)) / self.sigma ** 2))

    def reinforce(self, paths):
        """
        Implementation of the policy gradient method episodic REINFORCE.

        Parameters:
        n_itr: number of gradient ascent updates
        N : number of trajectory/episode
        T : length of each trajectory/episode

        Returns:
        alpha: the current optimum value of theta, argmin of J(theta)
        hess: the estimate of the hessian used in the PG-ELLA algorithm
        """

        hess = 0
        alpha = np.random.randn(self.obs_space_shape)
        # Theta update
        for episode in paths:
            r = np.array([(self.discount ** t) * episode['rewards'][t] for t in range(self.horizon)])
            g = np.array([self.gaussian_grad(episode['actions'][t],
                                             episode['states'][t], alpha) for t in range(self.horizon)])
            alpha += np.dot(r, g)
            hess = sum([np.outer(episode['states'][t], episode['states'][t].T) for t in range(self.horizon)])
            hess = hess / self.sigma ** 2
        alpha = alpha / self.nb_episodes
        hess = hess / self.nb_episodes

        return alpha, hess

    def pg_ella(self, nb_itr=5):

        """
        Implementation of the online multi-task learning PG-ELLA algorithm.

        Parameters:
        k: dimension of the vector s
        d: dimension of the vector theta
        s: (sparse) task-specific vector of coefficients
        lmbda: L1 penalization for matrix L hyper-parameter
        alpha: current optimum computed with REINFORCE algorithm
        hess: current hessian computed with REINFORCE algorithm
        nb_iter: number of iterations of the algorithm

        Returns:
        theta_opt: the optimal learned policy
        """
        d = self.obs_space_shape
        k = self.nb_tasks
        A = np.zeros((k * d, k * d))
        b = np.zeros((k * d, 1))
        L = np.zeros((d, k))
        s = np.random.randn(k, 1)
        T = defaultdict(dict)
        Q = {"task1": self.task1, "task2": self.task2, "task3": self.task3}

        for string, task in Q.items():
            T[str(string)]["count"] = 0
        for itr in range(nb_itr):
            for string, task in Q.items():

                if T[str(string)]["count"] > 5:
                    break

                if T[str(string)]["count"] == 0:
                    random_policy = Policy(random=True)
                    paths = self.collect_episodes(task, policy=random_policy, horizon=self.horizon,
                                                  n_episodes=self.nb_episodes)
                else:
                    hess = T[str(string)]["hess"]
                    alpha = T[str(string)]["alpha"]
                    s = T[str(string)]["s"]

                    best_pol = Policy(alpha, self.sigma, self.action_space_shape)
                    paths = self.collect_episodes(task, policy=best_pol, horizon=self.horizon,
                                                  n_episodes=self.nb_episodes)
                    A = A - np.kron(np.outer(s, s.T), hess)
                    assert A.shape == (k * d, k * d)
                    temp = np.kron(s.T, np.dot(alpha.T, hess))
                    b = b - temp.reshape(-1, 1)
                    assert b.shape == (k * d, 1)

                new_alpha, new_hess = self.reinforce(paths)
                T[str(string)]["hess"] = new_hess
                T[str(string)]["alpha"] = new_alpha

                L = reinitialize_zero_columns(L)
                fun = lambda x: loss(L, x, new_alpha, new_hess)
                new_s = minimize(fun, s).x
                new_s = np.expand_dims(new_s, 1)
                T[str(string)]["s"] = new_s
                assert new_s.shape == (k, 1)
                A = A + np.kron(np.outer(new_s, new_s.T), new_hess)
                A = (1 / self.nb_tasks) * A
                assert A.shape == (k * d, k * d)
                temp = np.kron(new_s.T, np.dot(new_alpha.T, new_hess))
                b = b + temp.reshape(-1, 1)
                b = (1 / self.nb_tasks) * b
                assert b.shape == (k * d, 1)
                L = np.dot(np.linalg.inv(A + self.lbda * np.eye(k * d, k * d)), b)
                L = L.reshape(d, k)
                theta_new = np.dot(L, new_s)
                assert theta_new.shape == (d, 1)
                T[str(string)]["theta"] = theta_new
                T[str(string)]["count"] += 1

        return T


class Policy(object):

    def __init__(self, theta=None, sigma=None, act_space=None, random=False):
        self.random = random
        self.theta = theta
        self.sigma = sigma
        self.act_space = act_space

    def draw_action(self, state, env):
        if self.random:
            return env.action_space.sample()
        else:
            assert state.shape == self.theta.shape
            return np.dot(self.theta.T, state) + self.sigma * np.random.randn(self.act_space)


if __name__ == "__main__":
    ag = Agent()
    ag.pg_ella()
