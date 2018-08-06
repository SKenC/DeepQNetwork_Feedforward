import numpy as np

class Emulator:
    """
    This class represents a video game.
    source : https://github.com/algolab-inc/tf-dqn-simple/blob/master/catch_ball.py
    """
    def __init__(self):
        self.actions = (0, 1, 2)
        self.action_num = len(self.actions)     #number of actions

        self.screen_x = 30
        self.screen_y = 30
        
        self.screen = np.zeros([self.screen_y, self.screen_x])

        self.player_len = 5

        self.reset()


    def update(self, action):
        """
        Args
            action: Enable actions of The Agent. 0:stay. 1:left, 2:right
        :return:

        """

        #UPDATE player position.
        if action == self.actions[1]:
            #left
            self.player_x = max(0, self.player_x - 1)
        elif action == self.actions[2]:
            #right
            self.player_x = min(self.player_x + 1, self.screen_x - self.player_len)
        else:
            pass

        #UPDATE ball position
        self.ball_y += 1

        #evaluate the reward
        self.reward = 0
        self.terminal = False
        if self.ball_y >= self.screen_y-1:
            self.terminal = True
            if self.player_x <= self.ball_x < self.player_x + self.player_len:
                #success
                self.reward = 1
            else :
                #miss
                self.reward = -1

    def draw(self):
        self.screen = np.zeros([self.screen_y, self.screen_x])

        self.screen[self.player_y, self.player_x:self.player_x+self.player_len] = 1

        self.screen[self.ball_y, self.ball_x] = 1
        
        return np.expand_dims(self.screen, axis=2)

    def observe(self):
        self.draw()
        return np.expand_dims(self.screen, axis=2), self.reward, self.terminal

    def step(self, action):
        self.update(action=action)
        self.observe()
        return np.expand_dims(self.screen, axis=2), self.reward, self.terminal, 0 #dummy

    def reset(self):
        self.reward = 0
        self.terminal = False

        self.player_y = self.screen_y - 1
        self.player_x = np.random.randint(self.screen_x - self.player_len)

        self.ball_y = 0
        self.ball_x = np.random.randint(self.screen_x)
        
    def render(self):
        return self.screen * 255
        





