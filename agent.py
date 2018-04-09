import numpy as np
from utils import *
import gym
from scipy.optimize import minimize
from utils import stringer
from tqdm import tqdm
import copy
from parameters import parameters
from collections import defaultdict


class Agent(object):
    """
    Online Learner able to learn efficiently several tasks.
    """

    def __init__(self, n_iter, nb_episodes, horizon):
        """

        :param n_iter: number of reinforce iteration for the gradient ascent of the reinforce algorithm
        :param nb_episodes: number of trajectories simulated to compute the expectation
        :param horizon: maximal length of each trajectory
        """

        self.env = gym.make(parameters["env_name"])  # Open AI gym environment
        self.n_iter = n_iter # Nb of iteration of the PG-ELLA for each task
        self.nb_tasks = 2  # Number of tasks
        self.discount = 0.99  # Discount rate
        self.sigma = 0.4  # Variance for the Gaussian stochastic policy
        self.learning_rate = 1e-8  # Learning rate for the gradient ascent
        self.obs_space_shape = self.env.observation_space.shape[0]
        self.action_space_shape = self.env.action_space.shape[0]
        self.nb_episodes = nb_episodes
        self.horizon = horizon
        self.lbda = 0.1
        self.mu = 0.5  # The L1-penalization hyperparameter for s

    def collect_episodes(self, mdp, policy=None, horizon=None, n_episodes=1, render=False, index=None,
                         other_value=None, test=False):
        """
        Takes an environment in input and returns the collected actions, rewards and observations of the
        simulation.
        Parameters:
        mdp -- Markov Decision Process (or environment)
        Returns:
        paths -- the collected path in a dictionnary form
        """

        paths = []
        action_old = other_value  # Given that we have two tasks (the two actions), we have to fix one each time to
        # only learn the one we are interested in

        for j in range(n_episodes):
            observations = []
            actions = []
            rewards = []
            next_states = []
            state = mdp.reset()

            for i in range(horizon):
                if test:
                    action = policy.draw_action(state, mdp)  # At test time, we want to draw the two actions at the
                    # same time
                else:
                    if index == 0:  # index corresponding to the first task
                        action = np.array([policy.draw_action(state, mdp), action_old])  # We fix the other task
                    else:
                        action = np.array([action_old, policy.draw_action(state, mdp)])  # Same for the second task
                next_state, reward, terminal, _ = mdp.step(action)
                if render:
                    mdp.render()  # To visualise the learned policy at test time
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
                next_states=np.array(next_states),
                average_reward=np.mean(rewards)
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

    def reinforce(self, paths, index, nb_iter=100):
        """
        Implementation of the policy gradient method episodic REINFORCE.

        Parameters:
        paths: the collected episodes
        index: the index for whether to update the task 0 or 1
        Returns:

        alpha: the current optimum value of theta, argmin of J(theta)
        hess: the estimate of the hessian used in the PG-ELLA algorithm
        """

        theta = np.random.randn(self.obs_space_shape)  # Initialize theta at random each time
        if index == 0:
            grad = lambda theta: - np.mean([(self.discount ** t) * episode['rewards'][t]
                                            * self.gaussian_grad(episode['actions'][t][0],
                                            episode['states'][t], theta) for t in range(self.horizon)
                                            for episode in paths])  # Corresponding to the gradient in 5.1 (article)
        # Note that the authors forgot to multiply by the state when computing the gradient of the Gaussian density
        else:
            grad = lambda theta: - np.mean([(self.discount ** t) * episode['rewards'][t]
                                            * self.gaussian_grad(episode['actions'][t][1],
                                            episode['states'][t], theta) for t in range(self.horizon)
                                            for episode in paths])
        for itr in range(nb_iter):
            theta += self.learning_rate * grad(theta)  # Gradient ascent to compute the best theta (alpha)

        hess = np.mean([(1 / self.sigma) * np.outer(episode['states'][t], episode['states'][t].T)
                        for t in range(self.horizon) for episode in paths], axis=0)  # Computing the hessian matrix
        # Following the formula in 5.1 from the authors
        alpha = theta

        return alpha, hess

    def pg_ella(self):
        """
        Implementation of the online multi-task learning PG-ELLA algorithm.
        Parameters:

        Returns:
        T: a dictionnary filled with the parameters corresponding to the optimal stochastic policy
        """
        # Initialization of all the parameters

        d = self.obs_space_shape
        k = self.nb_tasks
        A = np.zeros((k * d, k * d))
        b = np.zeros((k * d, 1))
        L = np.zeros((d, k))
        s = np.random.randn(k, 1)
        T = defaultdict(dict)
        env = self.env

        # The two tasks in this environment correspond to [main engine, left-right engines].
        # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
        # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off

        Q = {"task1_x": env, "task1_y": env}

        for string, task in Q.items():
            T[string]["count"] = 0  # Number of iteration for each task
            T[string]["last_value"] = 0  # Keeping the last value of the action
            # to keep it fixed while training the other
            if "x" in string:
                T[string]["index"] = 0
            else:
                T[string]["index"] = 1

        for itr in tqdm(range(self.n_iter)):
            for string, task in Q.items():  # Looping over the tasks
                string_ = stringer(string)  # string_ is the other task

                if T[str(string)]["count"] == 0:  # Corresponds to if isNewTask in the algorithm
                    random_policy = Policy(T[string], random=True) # Runing a random policy
                    paths = self.collect_episodes(task, policy=random_policy, horizon=self.horizon,
                                                  n_episodes=self.nb_episodes, index=T[string]["index"],
                                                      other_value=T[string_]["last_value"])  # Collecting Trajectories
                    T[string]["last_value"] = paths[-1]["actions"][-1][T[string]["index"]]  # Saving the last value
                else:
                    hess = T[string]["hess"]  # If the task was already trained, we retrieve the hessian, alpha and s
                    alpha = T[str(string)]["alpha"]
                    s = T[str(string)]["s"]

                    best_pol = Policy(T[string], self.sigma)  # The best policy using the last learned parameters
                    paths = self.collect_episodes(task, policy=best_pol, horizon=self.horizon,
                                                      n_episodes=self.nb_episodes, index=T[string]["index"],
                                                      other_value=T[string_]["last_value"])  # Collecting the episodes

                    print("Average Reward for iteration {} and taks {}".format(itr, string[-1]),
                    np.mean([episode['average_reward'] for episode in paths]))

                    T[string]["last_value"] = paths[-1]["actions"][-1][T[string]["index"]]  # Updating the
                    # last action value
                    A = A - np.kron(np.dot(s, s.T), hess)
                    assert A.shape == (k * d, k * d)
                    temp = np.kron(s.T, np.dot(alpha.T, hess))
                    b = b - temp.reshape(-1, 1)
                    assert b.shape == (k * d, 1)

                new_alpha, new_hess = self.reinforce(paths, T[string]["index"])  # Computing alpha and hessian with
                # REINFORCE algorithm
                T[str(string)]["hess"] = new_hess  # Updating the parameters
                T[str(string)]["alpha"] = new_alpha
                new_alpha = np.expand_dims(new_alpha, 1)
                L = reinitialize_zero_columns(L)
                fun = lambda x: loss(L, x, new_alpha, new_hess, self.mu)
                new_s = minimize(fun, s).x  # Using scipy minimize to find the best s
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
                theta_new = np.squeeze(theta_new)  # Computing the new theta for this task/action
                assert theta_new.shape == (d,)
                T[str(string)]["theta"] = theta_new
                T[str(string)]["count"] += 1
        return T

    def reinforce_w_pg_ella(self):
        """
        Function used to visualize the effectiveness of the learning process
        """

        T = self.pg_ella()
        best_pol = Policy(T, self.sigma, test=True)
        self.collect_episodes(self.env, policy=best_pol, horizon=150,
                                      n_episodes=10, render=True, test=True)
        self.env.close()


class Policy(object):
    """
    Policy class with draw_action method to sample actions with the current stochastic policy.
    """

    def __init__(self, actions_dict=None, sigma=None, random=False, test=False):
        """
        :param actions_dict: the dictionnary filled with the parameters
        :param sigma: the variance needed for the Gaussian policy
        :param random: to get random episodes
        :param test: if test is true, we want to use both best parameters for the first and second task
        """

        self.random = random
        self.sigma = sigma
        self.actions_dict = actions_dict
        self.test = test

    def draw_action(self, state, env, ):
        """
        Takes as input the current state and the environment.
        Parameters:
        state -- the current state
        env -- the current environment

        Returns:
        action -- the sampled action resulting from the current parametric stochastic policy
        """

        if self.test:
            action_x = np.dot(self.actions_dict["task1_x"]["theta"].T, state) + self.sigma * np.random.randn()
            action_x = cap_action(action_x) # Cap the action between - 1 and 1
            action_y = np.dot(self.actions_dict["task1_y"]["theta"].T, state) + self.sigma * np.random.randn()
            action_y = cap_action(action_y)
            return np.array([action_x, action_y])
        if self.random:
            action = env.action_space.sample()[self.actions_dict["index"]]
            return action
        else:
            action = np.dot(self.actions_dict["alpha"].T, state) + self.sigma * np.random.randn()
            action = cap_action(action)  # Cap the action between - 1 and 1
            return action


# - Stacker toutes les trajectoires et apprendre alpha, hess, s, A, b et L sur cette longue séquence d’évènements ou
# sélectionner successivement chacune des trajectoires et pour chacune d’elle calculer alpha, hess,
# puis mettre à jour s, A, b et L

# - Pour une tâche déjà rencontrée, on va resimuler le même nombre de trajectoires.
# Il faudrait, pour chacune d’elle, repartir de la trajectoire simulée la dernière fois que la tâce a été rencontrée
# (la première trajectoire de notre seconde tâche i va partir du dernier state, et simuler sur tout son
# horizon selon le alpha estimé, de la première trajectoire de notre première tâche i)




