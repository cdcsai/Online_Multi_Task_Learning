import numpy as np
from utils import reinitialize_zero_columns, discounted_r, loss, hess_norm, stringer
import gym
from gym_extensions.continuous import gym_navigation_2d
from scipy.optimize import minimize
from copy import deepcopy
import copy
from parameters import parameters

from tqdm import tqdm
from collections import defaultdict


class Agent2(object):
    """
    Online Learner able to learn efficiently several tasks.
    """

    def __init__(self):

        self.env = gym.make(parameters["env_name2"])
        self.render = False
        self.n_iter = 2
        self.nb_tasks = 2
        self.discount = 0.9
        self.sigma = 1
        self.learning_rate = 0.001
        self.obs_space_shape = self.env.observation_space.shape[0]
        self.action_space_shape = self.env.action_space.shape[0]
        self.theta1 = np.random.randn(self.obs_space_shape)
        self.theta2 = np.random.randn(self.obs_space_shape)
        self.nb_episodes = 10
        self.horizon = 5
        self.lbda = 0.1

    @property
    def create_multitask_envs(self):
        env1 = self.env
        env2 = self.env
        env3 = self.env

        env1.env.destination = np.array([20, 100])
        env2.env.destination = np.array([300, 350])
        env3.env.destination = np.array([500, 500])
        if self.render:
            env1.render()
            env2.render()
            env3.render()
        return env1, env2, env3

    def collect_episodes(self, mdp, policy=None, horizon=None, n_episodes=1, render=False, index=None,
                         other_value=None, test=False):

        """
        Takes a Markov Decision Process in input and returns the collected actions, rewards and observations of the
        simulation.

        Parameters:
        mdp -- Markov Decision Process (or environment)

        Returns:
        paths -- the collected path in a dictionnary form
        """

        paths = []
        action_old = other_value

        for _ in range(n_episodes):
            observations = []
            actions = []
            rewards = []
            next_states = []

            state = mdp.reset()
            # state = state.reshape(-1, 1)

            for _ in range(horizon):

                if not test:
                    if index == 0:
                        action = np.array([policy.draw_action(state, mdp), action_old])
                    else:
                        action = np.array([action_old, policy.draw_action(state, mdp)])


                else:
                    action = policy.draw_action(state, mdp)
                action = action / np.linalg.norm(action)
                next_state, reward, terminal, _ = mdp.step(action)
                if render:
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

    def gaussian_grad(self, action, state, theta, index):
        """
        Compute the gradient of the gaussian policy. Used in the reinforce policy gradient method.

        Parameters:
        current action, current state, current theta

        Returns:
        The updated gradient
        """

        return np.squeeze(np.dot(state, (action[index] - np.dot(theta.T, state)) / self.sigma ** 2))

    def reinforce(self, paths, index):
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
        lr = self.learning_rate
        for episode in paths:
            lr /= 10
            for t in range(self.horizon):
                r = (self.discount ** t) * episode['rewards'][t]
                alpha += r * lr * self.gaussian_grad(episode['actions'][t],
                                                                     episode['states'][t], alpha, index)

            hess = sum([np.outer(episode['states'][t], episode['states'][t].T) for t in range(self.horizon)])
            hess = hess / self.sigma ** 2
        alpha = -(alpha / self.nb_episodes)
        hess = (hess / self.nb_episodes)

        return alpha, hess

    def pg_ella(self):

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
        Q = {"task1_x": self.env, "task1_y": self.env}

        for string, task in Q.items():
            T[string]["count"] = 0
            T[string]["last_value"] = 0
            if "x" in string:
                T[string]["index"] = 0
            else:
                T[string]["index"] = 1

        for itr in range(self.n_iter):
            for i, (string, task) in zip(range(1, len(Q)+1), Q.items()):
                string_ = stringer(string)
                print(T[str(string)]["count"], T[str(string_)]["count"])
                if T[string]["count"] == 0:
                        random_policy = Policy(T[string], random=True)
                        paths = self.collect_episodes(task, policy=random_policy, horizon=self.horizon,
                                                      n_episodes=self.nb_episodes, index=T[string]["index"],
                                                      other_value=T[string_]["last_value"])
                        T[string]["last_value"] = paths[-1]["next_states"][-1][T[string]["index"]]
                else:
                    hess = T[string]["hess"]
                    alpha = T[str(string)]["alpha"]
                    s = T[str(string)]["s"]

                    best_pol = Policy(T[string], self.sigma)
                    paths = self.collect_episodes(task, policy=best_pol, horizon=self.horizon,
                                                  n_episodes=self.nb_episodes, index=T[string]["index"],
                                                  other_value=T[string_]["last_value"])

                    T[string]["last_value"] = paths[-1]["next_states"][-1][T[string]["index"]]

                    A = A - np.kron(np.outer(s, s.T), hess)
                    assert A.shape == (k * d, k * d)
                    temp = np.kron(s.T, np.dot(alpha.T, hess))
                    b = b - temp.reshape(-1, 1)
                    assert b.shape == (k * d, 1)

                new_alpha, new_hess = self.reinforce(paths, T[string]["index"])
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

    def reinforce_w_pg_ella(self):
        """

        :param task:
        :return:
        """
        T = self.pg_ella()
        best_theta1, best_theta2 = T["task1_x"]["theta"], T["task1_y"]["theta"]
        best_pol = Policy_2(best_theta1, best_theta2, self.sigma)
        self.collect_episodes(self.env, policy=best_pol, horizon=self.horizon,
                                      n_episodes=self.nb_episodes, render=True, test=True)

    def reinforce_wo_pg_ella(self):
        """
        :param task:
        :return:
        """
        random_policy = Policy(random=True)
        paths = self.collect_episodes(self.task1, policy=random_policy, horizon=self.horizon,
                                          n_episodes=self.nb_episodes)

        best_theta1, best_theta2 = self.reinforce(paths), self.reinforce(paths)
        best_pol = Policy_2(best_theta1, best_theta2, self.sigma, self.action_space_shape)
        self.collect_episodes(self.task1, policy=best_pol, horizon=self.horizon,
                                          n_episodes=self.nb_episodes, render=True)

class Policy(object):

    def __init__(self, actions_dict=None, sigma=None, random=False):
        self.random = random
        self.sigma = sigma
        self.actions_dict = actions_dict

    def draw_action(self, state, env):
        if self.random:
            return env.action_space.sample()[self.actions_dict["index"]]
        else:
            return np.float(np.dot(self.actions_dict["alpha"].T, state) + self.sigma * np.random.randn())


class Policy_2(object):

    def __init__(self, theta1, theta2, sigma=None, random=False):
        self.random = random
        self.sigma = sigma
        self.theta1 = theta1
        self.theta2 = theta2

    def draw_action(self, state, env):
        action_x = round(np.float(np.dot(self.theta1.T, state) + self.sigma * np.random.randn()))
        action_y = round(np.float(np.dot(self.theta2.T, state) + self.sigma * np.random.randn()))
        return action_x, action_y


if __name__ == "__main__":
    ag = Agent2()
    ag.pg_ella()
    ag.reinforce_w_pg_ella()
    #ag.reinforce_wo_pg_ella()
