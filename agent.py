import numpy as np
from utils import reinitialize_zero_columns, discounted_r, loss, hess_norm
import gym
from gym_extensions.continuous import gym_navigation_2d
from scipy.optimize import minimize
import copy
from collections import defaultdict


class Agent(object):
    """
    Online Learner able to learn efficiently several tasks.
    """

    def __init__(self):

        self.env = gym.make("State-Based-Navigation-2d-Map1-Goal1-v0")
        self.task1 = self.create_multitask_envs[0]
        self.task2 = self.create_multitask_envs[1]
        self.task3 = self.create_multitask_envs[2]
        self.nb_tasks = 3
        self.discount = 0.9
        self.alpha = 0.1
        self.sigma = 0.5
        self.learning_rate = 0.001
        self.stepper = ConstantStep(self.learning_rate)
        self.render = False
        self.obs_space_shape = self.env.observation_space.shape[0]
        self.action_space_shape = self.env.action_space.shape[0]
        self.theta = np.random.randn(self.obs_space_shape)
        self.alpha_dict = defaultdict(list)
        self.hess_dict = defaultdict(list)
        self.nb_episodes = 2
        self.horizon = 30
        self.lbda = 0.1

    @property
    def create_multitask_envs(self):
        env1 = self.env.env.destination = np.array([20, 100])
        env2 = self.env.env.destination = np.array([100, 150])
        env3 = self.env.env.destination = np.array([500, 500])
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
                action = policy.draw_action(state)
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

        return np.squeeze(np.outer(state, (np.linalg.norm(action, ord=2) - np.dot(theta.T, state)) / self.sigma ** 2))

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
        # Theta update
        for episode in paths:
            for t in range(1, self.horizon - 1):
                self.theta += (self.discount ** t) * episode['rewards'][t] * \
                              self.stepper.update(self.gaussian_grad(episode['actions'][t], episode['states'][t],
                                                                     self.theta))
                hess += np.outer(episode['states'][t], episode['states'][t].T) / self.sigma ** 2
                self.theta = self.theta / self.nb_episodes
                hess = hess / self.nb_episodes
        alpha = self.theta

        return alpha, hess

    def pg_ella(self) -> object:

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
        Q = [self.task1, self.task2, self.task3]

        for task in Q:

            if len(T[str(task)]) > 5:
                break

            if len(T[str(task)]) == 0:
                random_policy = Policy(random=True)
                paths = self.collect_episodes(task, policy=random_policy, horizon=self.horizon,
                                              n_episodes=self.nb_episodes)
            else:
                hess = T[str(task)]["hess"]
                alpha = T[str(task)]["alpha"]

                best_pol = Policy(alpha, self.sigma, self.action_space_shape)
                paths = self.collect_episodes(self.env, policy=best_pol, horizon=self.horizon,
                                              n_episodes=self.nb_episodes)
                A = A - np.kron(np.dot(s, s.T), hess)
                temp = np.kron(s.T, np.dot(alpha.T, hess))
                b = b - temp.reshape(-1, 1)

            alpha, hess = self.reinforce(paths)

            L = reinitialize_zero_columns(L)

            fun = lambda x: loss(L, x, alpha, hess)
            s = minimize(fun, s)
            A = A + np.kron(np.dot(s, s.T), hess)
            A = (1 / self.horizon) * A
            temp = np.kron(s.T, np.dot(alpha, hess))
            b = b + temp.reshape(-1, 1)
            b = (1 / self.horizon) * b
            L = np.dot(np.linalg.inv(A + self.lbda * np.eye(k * d, k * d)), b)
            L = L.reshape(d, k)

        theta_opt = np.dot(L, s)

        return theta_opt


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


class ConstantStep(object):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, gt):
        return self.learning_rate * gt


if __name__ == "__main__":
    ag = Agent()
    ag.pg_ella()
