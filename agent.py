import numpy as np
from sporco.admm import bpdn
from utils import reinitialize_zero_columns, discounted_r
import gym
import copy
from parameters import parameters


class Agent(object):
    """
    Online Learner able to learn efficiently several tasks.
    """

    def __init__(self, env, data_path):

        self.parameters = getattr(parameters, env)
        self.env = gym.make(self.parameters["env_name"])
        self.data_path = data_path
        self.discount = 0.9
        self.alpha = 0.1
        self.sigma = 0.5
        self.stepper = ConstantStep(0.001)

    @ staticmethod
    def collect_episodes(mdp, policy=None, horizon=None, n_episodes=1):

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
            for _ in range(horizon):
                action = policy.draw_action(state)
                next_state, reward, terminal, _ = mdp.step(action)
                # env.render()
                observations.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                state = copy.copy(next_state)
                if terminal:
                    # Finish rollout if terminal state reached
                    break
                    # We need to compute the empirical return for each time step along the
                    # trajectory

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

        return action - np.dot(theta, state) / self.sigma ** 2

    def reinforce(self, n_itr, N=100, T=20):
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

        theta = np.random.randn()
        ret = 0
        hess = 0
        for _ in range(n_itr):
            gaussian_policy = Policy(theta, self.sigma)
            paths = Agent.collect_episodes(self.env, policy=gaussian_policy, horizon=T, n_episodes=N)
            # Theta update
            for episode in paths:
                for t in range(T):
                    ret += (self.discount ** t) * episode['rewards'][t] * \
                           self.stepper.update(Agent.gaussian_grad(episode['actions'][t], episode['states'][t], theta))
                    hess += episode['states'][t] ** 2 / self.sigma ** 2
            theta += ret / N
            hess += hess / N
        alpha = theta

        return alpha, hess

    def pg_ella(self, k, d, s, t, alpha, hess, dims, lmbda, nb_iter=100):

        """
        Implementation of the online multi-task learning PG-ELLA algorithm.

        Parameters:
        k: dimension of the vector s
        d: dimension of the vector theta
        s: (sparse) task-specific vector of coefficients
        lmbda: L1 penalization for matrix L hyper-parameter
        alpha: current optimum computed with REINFORCE algorithm
        hess: current hessiane computed with REINFORCE algorithm
        nb_iter: number of iterations of the algorithm

        Returns:
        theta_opt: the optimal learned policy
        """

        A = np.zeros((k * d, k * d))
        b = np.zeros((k * d, 1))
        L = np.zeros((d, k))
        T = []

        for itr in range(1, nb_iter):
            if t not in T:
                (T, R) = Agent.collect_episodes(self.env)
            else:
                best_pol = Policy(alpha, self.sigma)
                (T, R) = Agent.collect_episodes(self.env, policy=best_pol)
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
            L = np.reshape(dims)

        theta_opt = np.dot(L, s)

        return theta_opt


class Policy(object):
    def __init__(self, theta, sigma):
        self.theta = theta
        self.sigma = sigma

    def draw_action(self, state):
        return np.random.normal(self.theta * state, self.sigma)


class ConstantStep(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, gt):
        return self.learning_rate * gt


if __name__ == "__main__":
    pass