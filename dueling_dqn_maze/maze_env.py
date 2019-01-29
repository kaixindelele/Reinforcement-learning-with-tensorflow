"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the environment part of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
"""
Reinforcement learning maze example
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = 1].
All other states:       ground      [reward = 0].
The observation is the dist between self.oval and self.rect
This script is the environment part of this example.
And I modified the script to custom hells.
The RL is in RL_brain.py.

"""

import numpy as np
import tkinter as tk
import time
import sys

# the pixel of pane
UNIT = 80
RECT_SIZE = 30
# the num of panes
MAZE_H = 7
MAZE_W = 7

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u','d','l','r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title("myMaze")
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT,MAZE_W*UNIT))
        self._built_maze()

    def create_hell(self,x,y):
        hell_center = self.origin + np.array([UNIT * x, UNIT * y])
        hell = self.canvas.create_rectangle(
            hell_center[0] - RECT_SIZE, hell_center[1] - RECT_SIZE,
            hell_center[0] + RECT_SIZE, hell_center[1] + RECT_SIZE,
            fill = 'black'
        )
        return hell

    def create_rect(self,x,y):
        center = self.origin + np.array([UNIT * x, UNIT * y])
        rect = self.canvas.create_rectangle(
            center[0] - RECT_SIZE, center[1] - RECT_SIZE,
            center[0] + RECT_SIZE, center[1] + RECT_SIZE,
            fill = 'red'
        )
        return rect

    def _built_maze(self):
        self.canvas = tk.Canvas(self,bg='white',
                                height = MAZE_H * UNIT,
                                width = MAZE_W * UNIT
                                )
    #     create grids
        for c in range(0,MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1, width = 1,fill = "yellow", tags = "line")
        for r in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1, width = 1,fill = "red", tags = "line")

        # create origin
        self.origin = np.array([UNIT/2,UNIT/2])

        # create active rect
        self.rect = self.create_rect(0,0)

        # create paradise
        oval_center = self.origin + np.array([UNIT * 3,UNIT* 4])
        self.oval = self.canvas.create_oval(
            oval_center[0] - RECT_SIZE, oval_center[1] - RECT_SIZE,
            oval_center[0] + RECT_SIZE, oval_center[1] + RECT_SIZE,
            fill = 'yellow'
        )

        # create hells
        self.hell_1 = self.create_hell(2,3)
        self.hell_2 = self.create_hell(4,2)

        # pick all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.03)
        self.canvas.delete(self.rect)
        self.rect = self.create_rect(0,0)
        self.update()
        time.sleep(0.3)
        # the observation is the dist between self.oval and self.rect
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)

    def step(self,action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:
            if s[1] > UNIT:
                base_action[1] -= UNIT
        if action == 1:
            if s[1] < MAZE_H * UNIT - UNIT:
                base_action[1] += UNIT
        if action == 2:
            if s[0] > UNIT:
                base_action[0] -= UNIT
        if action == 3:
            if s[0] < MAZE_W * UNIT -UNIT:
                base_action[0] += UNIT
        # base_action[0] control the horizontal direction moving
        # base_action[1] control the vertical direction moving
        self.canvas.move(self.rect,base_action[0],base_action[1])

        # next_state
        next_state = self.canvas.coords(self.rect)
        if next_state == self.canvas.coords(self.oval):
            reward = 1
            done = True
        elif next_state in [self.canvas.coords(self.hell_1) , self.canvas.coords(self.hell_2)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False
        # dist was normalized
        s_ = (np.array(next_state[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/UNIT/MAZE_H
        self.render()
        return s_,reward,done

    def render(self):
        self.update()
        time.sleep(0.001)

def main():
    maze = Maze()
    maze.reset()
    done = False
    # while not done:
    for i in range(100):
        action = np.random.randint(0,4,1)
        print(action)
        obs, reward, done = maze.step(action)
        print(obs,reward,done)
    maze.reset()

if __name__=="__main__":
    main()