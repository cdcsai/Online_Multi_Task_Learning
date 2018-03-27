import numpy as np
import tensorflow as tf
import gym
import parameters

class Agent(object):

    def __init__(self, env, data_path):

        self.parameters = getattr(parameters, env)
        self.env = gym.make(self.parameters["env_name"])
        self.data_path = data_path


    def enac_policy(self, state, sigma, theta):
        mu = np.dot(theta.T, state)
        action = sigma * np.random.randn() + mu
        return action

    def run_policy(self, num_episodes=1000, gamma=0.95, T_max=200):
        returns_sum = np.zeros(11)
        returns_count = np.zeros(11)
        # The final value function
        V = np.zeros(11)
        V_history = list()
        for e in range(1, num_episodes + 1):

            # Generate an episode.
            episode = []
            state = env.reset()
            for t in range(T_max):
                action = policy(state)
                next_state, reward, done, = env.step(state, action)
                episode.append((state, action, reward))
                if done:
                    break
                state = next_state

            states_in_episode = set(x[0] for x in episode)

            for state in states_in_episode:
                first_occurence_idx = next(i for i, x in enumerate(episode) if x[0] == state)
                G = sum([x[2] * (gamma ** i) for i, x in enumerate(episode[first_occurence_idx:])])
                returns_sum[state] += G
                returns_count[state] += 1.0
                V[state] = returns_sum[state] / returns_count[state]
            V_history.append(V.copy())
        return V, action

    def make_pgq(n):
        q = q_learning(n)

        def pi_greedy_q(state):
            a = np.argmax(q[state])
            return a

        return pi_greedy_q



    def play(self):
        env = 0
        pass


