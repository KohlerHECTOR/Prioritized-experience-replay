from mazemdp.toolbox import softmax, egreedy
import numpy as np
from mazemdp.mdp import Mdp
"""
Author : Hector Kohler
"""
class DynaQAgent():
    """
    A Dyna-Q agent based on Sutton and Barto Intro to RL, 2nd Ed, 2014-2015, figure 8.4 .
    """
    def __init__(self,
    mdp: Mdp,
    alpha: float = 0.1,
    gamma: float = 0.95
):
        self.mdp = mdp
        self.mdp.gamma = gamma
        self.alpha = alpha
        self.Q = np.zeros((self.mdp.nb_states, self.mdp.action_space.size))
        self.actions = np.arange(self.mdp.action_space.size)
        self.states = np.arange(self.mdp.nb_states)
        self.Model = {} # empty dictionary a model will look like this: Model[(S,A)] = (R, S')
        self.visited_states = {} # visited states during real experiments vis_states[S] = [False, False, True, False]


    def evaluate(self, eps, seed):
        """
        simple function to evaluate a policy from a Q table
        """
        self.mdp.reset()
        self.mdp.current_state = 0
        s = self.mdp.current_state
        done = self.mdp.done()

        np.random.seed(seed)
        steps_to_exit = 0
        while not done:


            a = egreedy(self.Q, s, eps)
            s, r, done, _  = self.mdp.step(a)

            steps_to_exit += 1

        return steps_to_exit


    def backup(self, s, a, r, s_next):
        """
        k-step Bellman backup
        """
        self.Q[s,a] = self.Q[s,a] + self.alpha * (self.target(s_next, a, r) - self.Q[s,a])

    def target(self, s_next, a, r):
        """
        1-step target
        """
        return r + self.mdp.gamma * np.max(self.Q[s_next,:])

    def update_model(self, s, a, r, s_next):
        """
        Model update: M(S, A) <-- R, S'
        """
        self.Model[(s, a)] = (r, s_next)

    def planning(self, eps):
        """
        Performs a planning step (to be repeated n times)
        """
        # choose a random visited state
        s = np.random.choice(list(self.visited_states.keys()))
        # choose a random action already performed in the visited state
        a = np.random.choice(self.actions)

        # while the sampled action has not been executed in the sample state, sample another action
        while self.visited_states[s][a] == False:
            a = np.random.choice(self.actions)

        # actual planning
        r, s_next = self.Model[(s, a)]
        # q_old = self.Q.copy()  #q_old and q_new are fore the gain computation

        # if s in self.mdp.terminal_states:
        #     self.Q[s, a] = self.alpha * r
        #
        # else:
        #     # Backup
        self.backup(s, a, r, s_next)

    def learn(self,
              eps: float, # epsilon parameter for eps-greedy pol
              nb_episodes: int = 50,
              n: int = 50, # planning steps
              render: bool = True,
              timeout: int = 50, # episode length
              seed: int = 42
             ):

        """
        Function implementing the algorihtm of Figure 8.4, Introduction to
        Reinforcement Learning, Sutton and barto, 2014, 2015 2nd Ed
        """

        self.mdp.timeout = timeout  # episode length
        list_steps_episode = []# to plot

        if render:
            self.mdp.new_render("Dyna-Q e-greedy")

        for _ in range(nb_episodes):

            s = self.mdp.reset()
            self.mdp.current_state = 0 #Sutton&Barto's simulation always starts from same state.
            s = self.mdp.current_state
            np.random.seed(seed)
            done = self.mdp.done()
            steps = 0
            while not done:

                # Update visited states
                self.visited_states[s] = [False] * self.mdp.action_space.size
                if render:
                    # Show the agent in the maze
                    self.mdp.render(self.Q, self.Q.argmax(axis=1))

                a = egreedy(self.Q, s, eps)
                # Update executed acions in s
                self.visited_states[s][a] = True
                # Execute action a
                s_next, r, done, _ = self.mdp.step(a)
                # q_old = self.Q.copy() # q_old and q_new are fore the gain computation

                # if s in self.mdp.terminal_states:
                #     self.Q[s, a] = self.alpha * r
                #
                # else:
                #     # Backup
                self.backup(s, a, r, s_next)



                # Update Model
                self.update_model(s, a, r, s_next)


                # Planning
                for _ in range(n):
                    self.planning(eps)

                # Update the agent position
                s = s_next


            steps_to_exit = self.evaluate(eps, seed)

            list_steps_episode.append(steps_to_exit)

        if render:
            # Show the final policy
            self.mdp.current_state = 0
            self.mdp.render(self.Q, np.argmax(self.Q, axis=1), title="Dyna-Q e-greedy")

        return list_steps_episode
