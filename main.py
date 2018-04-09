from agent import Agent
from parameters import parameters


def main():
    ag = Agent(parameters["n_iter"], parameters["nb_episodes"], parameters["horizon"])
    ag.reinforce_w_pg_ella()


if __name__ == "__main__":
    main()
