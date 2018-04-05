import numpy as np
from utils import reinitialize_zero_columns, discounted_r, loss, hess_norm
import gym
from gym_extensions.continuous import gym_navigation_2d
from scipy.optimize import minimize
from copy import deepcopy
import copy
from tqdm import tqdm
from collections import defaultdict


class Agent(object):
    """
    Online Learner able to learn efficiently several tasks.
    """

    def __init__(self):

        self.env = gym.make("State-Based-Navigation-2d-Map1-Goal1-v0")
        self.render = False
        self.task1, self.task2, self.task3 = self.create_multitask_envs
        self.n_iter = 10
        self.nb_tasks = 3
        self.discount = 0.9
        self.sigma = 3
        self.learning_rate = 0.01
        self.obs_space_shape = self.env.observation_space.shape[0]
        self.action_space_shape = self.env.action_space.shape[0]
        self.theta1 = np.random.randn(self.obs_space_shape)
        self.theta2 = np.random.randn(self.obs_space_shape)
        self.nb_episodes = 600
        self.horizon = 50
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

    def collect_episodes(self, mdp, policy=None, horizon=None, n_episodes=1, render=False):

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

    def gaussian_grad(self, action, state, theta):
        """
        Compute the gradient of the gaussian policy. Used in the reinforce policy gradient method.

        Parameters:
        current action, current state, current theta

        Returns:
        The updated gradient
        """

        return np.squeeze(np.dot(state, (action - np.dot(theta.T, state)) / self.sigma ** 2))

    def reinforce(self, paths, x=True):
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
            print(r)

            if x:
                g = np.array([self.gaussian_grad(episode['actions'][t][0],
                                                 episode['states'][t], alpha) for t in range(self.horizon)])

            else:
                g = np.array([self.gaussian_grad(episode['actions'][t][1],
                                                     episode['states'][t], alpha) for t in range(self.horizon)])

            alpha += np.dot(r, g)
            hess = sum([np.outer(episode['states'][t], episode['states'][t].T) for t in range(self.horizon)])
            hess = hess / self.sigma ** 2
        alpha = -(alpha / self.nb_episodes)
        hess = hess / self.nb_episodes

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
        T1 = defaultdict(dict)
        T2 = defaultdict(dict)
        Q = {"task1_x": self.task1, "task1_y": self.task1, "task2_x": self.task2, "task2_y": self.task2,
             "task3_x": self.task3, "task3_y": self.task3}

        for string, task in Q.items():
            T1[str(string)]["count"] = 0
            T2[str(string)]["count"] = 0

        for itr in range(self.n_iter):
            for i, (string, task) in zip(range(1, len(Q)+1), Q.items()):

                if T1[str(string)]["count"] > 5 or T2[str(string)]["count"] > 5:
                    break

                if T1[str(string)]["count"] == 0 or T2[str(string)]["count"] == 0:
                    random_policy = Policy(random=True)
                    paths = self.collect_episodes(task, policy=random_policy, horizon=self.horizon,
                                                  n_episodes=self.nb_episodes)
                else:
                    hess = T1[str(string)]["hess"]
                    alpha1 = T1[str(string)]["alpha1"]
                    alpha2 = T2[str(string)]["alpha2"]
                    s1 = T1[str(string)]["s1"]
                    s2 = T2[str(string)]["s2"]
                    best_pol = Policy(alpha1, alpha2, self.sigma, self.action_space_shape)
                    paths = self.collect_episodes(task, policy=best_pol, horizon=self.horizon,
                                                  n_episodes=self.nb_episodes)

                    if i % 2 != 0:
                        s = deepcopy(s1)
                        alpha = deepcopy(alpha1)
                    else:
                        s = deepcopy(s2)
                        alpha = deepcopy(alpha2)

                    A = A - np.kron(np.outer(s, s.T), hess)
                    assert A.shape == (k * d, k * d)
                    temp = np.kron(s.T, np.dot(alpha.T, hess))
                    b = b - temp.reshape(-1, 1)
                    assert b.shape == (k * d, 1)

                if i % 2 != 0:
                    new_alpha1, new_hess = self.reinforce(paths, x=True)
                    T1[str(string)]["hess"] = new_hess
                    T1[str(string)]["alpha1"] = new_alpha1
                    L = reinitialize_zero_columns(L)
                    fun = lambda x: loss(L, x, new_alpha1, new_hess)
                    new_s1 = minimize(fun, s).x
                    new_s1 = np.expand_dims(new_s1, 1)
                    T1[str(string)]["s1"] = new_s1
                    assert new_s1.shape == (k, 1)
                    A = A + np.kron(np.outer(new_s1, new_s1.T), new_hess)
                    A = (1 / self.nb_tasks) * A
                    assert A.shape == (k * d, k * d)
                    temp = np.kron(new_s1.T, np.dot(new_alpha1.T, new_hess))
                    b = b + temp.reshape(-1, 1)
                    b = (1 / self.nb_tasks) * b
                    assert b.shape == (k * d, 1)
                    L = np.dot(np.linalg.inv(A + self.lbda * np.eye(k * d, k * d)), b)
                    L = L.reshape(d, k)
                    theta1_new = np.dot(L, new_s1)
                    assert theta1_new.shape == (d, 1)
                    T1[str(string)]["theta1"] = theta1_new
                    T1[str(string)]["count"] += 1

                else:
                    new_alpha2, new_hess = self.reinforce(paths, x=False)
                    T2[str(string)]["alpha2"] = new_alpha2
                    L = reinitialize_zero_columns(L)
                    fun = lambda x: loss(L, x, new_alpha2, new_hess)
                    new_s2 = minimize(fun, s).x
                    new_s2 = np.expand_dims(new_s2, 1)
                    T2[str(string)]["s2"] = new_s2
                    assert new_s2.shape == (k, 1)
                    A = A + np.kron(np.outer(new_s2, new_s2.T), new_hess)
                    A = (1 / self.nb_tasks) * A
                    assert A.shape == (k * d, k * d)
                    temp = np.kron(new_s2.T, np.dot(new_alpha2.T, new_hess))
                    b = b + temp.reshape(-1, 1)
                    b = (1 / self.nb_tasks) * b
                    assert b.shape == (k * d, 1)
                    L = np.dot(np.linalg.inv(A + self.lbda * np.eye(k * d, k * d)), b)
                    L = L.reshape(d, k)
                    theta2_new = np.dot(L, new_s2)
                    assert theta2_new.shape == (d, 1)
                    T2[str(string)]["theta2"] = theta2_new
                    T2[str(string)]["count"] += 1

        return T1, T2

    def reinforce_w_pg_ella(self, task=1):
        """

        :param task:
        :return:
        """
        T1, T2 = self.pg_ella()
        if task == 1:
            best_theta1, best_theta2 = T1["task1_x"]["theta1"], T2["task1_y"]["theta2"]
            best_pol = Policy(best_theta1, best_theta2, self.sigma, self.action_space_shape)
            self.collect_episodes(self.task1, policy=best_pol, horizon=self.horizon,
                                      n_episodes=self.nb_episodes, render=True)

        elif task == 2:
            best_theta1, best_theta2 = T1["task2_x"]["theta1"], T2["task2_y"]["theta2"]
            best_pol = Policy(best_theta1, best_theta2, self.sigma, self.action_space_shape)
            self.collect_episodes(self.task2, policy=best_pol, horizon=self.horizon,
                                  n_episodes=self.nb_episodes, render=True)

        else:
            best_theta1, best_theta2 = T1["task3_x"]["theta1"], T2["task3_y"]["theta2"]
            best_pol = Policy(best_theta1, best_theta2, self.sigma, self.action_space_shape)
            self.collect_episodes(self.task3, policy=best_pol, horizon=self.horizon,
                                  n_episodes=self.nb_episodes, render=True)

    def reinforce_wo_pg_ella(self, task=2):
        """
        :param task:
        :return:
        """
        if task == 1:
            random_policy = Policy(random=True)
            paths = self.collect_episodes(self.task1, policy=random_policy, horizon=self.horizon,
                                          n_episodes=self.nb_episodes)

            best_theta1, best_theta2 = self.reinforce(paths, x=True)[0], self.reinforce(paths, x=False)[1]
            best_pol = Policy(best_theta1, best_theta2, self.sigma, self.action_space_shape)
            self.collect_episodes(self.task1, policy=best_pol, horizon=self.horizon,
                                          n_episodes=self.nb_episodes, render=True)
        elif task == 2:
            random_policy = Policy(random=True)
            paths = self.collect_episodes(self.task2, policy=random_policy, horizon=self.horizon,
                                          n_episodes=self.nb_episodes)
            best_theta1, best_theta2 = self.reinforce(paths, x=True)[0], self.reinforce(paths, x=False)[1]
            best_pol = Policy(best_theta1, best_theta2, self.sigma, self.action_space_shape)
            self.collect_episodes(self.task2, policy=best_pol, horizon=self.horizon,
                                  n_episodes=self.nb_episodes, render=True)

        else:
            random_policy = Policy(random=True)
            paths = self.collect_episodes(self.task3, policy=random_policy, horizon=self.horizon,
                                          n_episodes=self.nb_episodes)
            best_theta1, best_theta2 = self.reinforce(paths, x=True)[0], self.reinforce(paths, x=False)[1]
            best_pol = Policy(best_theta1, best_theta2, self.sigma, self.action_space_shape)
            self.collect_episodes(self.task3, policy=best_pol, horizon=self.horizon,
                                  n_episodes=self.nb_episodes, render=True)


class Policy(object):

    def __init__(self, theta1=None, theta2=None, sigma=None, random=False):
        self.random = random
        self.theta1 = theta1
        self.theta2 = theta2
        self.sigma = sigma

    def draw_action(self, state, env):
        if self.random:
            return env.action_space.sample()
        else:
            action_x = np.dot(self.theta1.T, state) + self.sigma * np.random.randn()
            action_y = np.dot(self.theta2.T, state) + self.sigma * np.random.randn()
            return action_x, action_y


if __name__ == "__main__":
    ag = Agent()
    #ag.reinforce_w_pg_ella()
    ag.reinforce_wo_pg_ella()
