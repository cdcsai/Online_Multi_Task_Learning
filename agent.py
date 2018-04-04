import numpy as np
from sporco.admm import bpdn
from utils import reinitialize_zero_columns, discounted_r
import gym
from gym_extensions.continuous import gym_navigation_2d
import copy


class Agent(object):

    """
    Online Learner able to learn efficiently several tasks.
    """

    def __init__(self):

        self.env = gym.make("State-Based-Navigation-2d-Map1-Goal1-v0")
        self.discount = 0.9
        self.alpha = 0.1
        self.sigma = 0.5
        self.learning_rate = 0.001
        self.stepper = ConstantStep(self.learning_rate)
        self.render = False
        self.obs_space_shape = self.env.observation_space.shape[0]
        self.action_space_shape = self.env.action_space.shape[0]
        self.theta = np.random.randn(self.obs_space_shape)

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
            #state = state.reshape(-1, 1)

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

    def reinforce(self, n_itr=1, N=50, T=10):
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

        ret = 0
        hess = 0
        for _ in range(n_itr):
            gaussian_policy = Policy(self.theta, self.sigma, self.action_space_shape)
            paths = self.collect_episodes(self.env, policy=gaussian_policy, horizon=T, n_episodes=N)
            # Theta update
            for episode in paths:
                for t in range(1, T-1):
                    self.theta += (self.discount ** t) * episode['rewards'][t] * \
                           self.stepper.update(self.gaussian_grad(episode['actions'][t], episode['states'][t],
                                                                  self.theta))
                    hess += np.outer(episode['states'][t], episode['states'][t].T) / self.sigma ** 2
            self.theta = self.theta / N
            hess = hess / N
        alpha = self.theta

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
        hess: current hessian computed with REINFORCE algorithm
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
                best_pol = Policy()
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
    
    def __init__(self, theta, sigma, act_space):
        self.theta = theta
        self.sigma = sigma
        self.act_space = act_space

    def draw_action(self, state):
        assert state.shape == self.theta.shape
        return np.dot(self.theta.T, state) + self.sigma * np.random.randn(self.act_space)


class ConstantStep(object):
    
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, gt):
        return self.learning_rate * gt


if __name__ == "__main__":
    ag = Agent()
    ag.reinforce()
