import tensorflow as tf
import emulator
import numpy as np

def simple_test(model, episode_num):
    """
    test the agent in the test environment episode_num times.
    """
    env = emulator.Emulator()

    win_num = 0
    lose_num = 0
    for episode in range(episode_num):
        #initialization
        env.reset()
        state, reward, terminal = env.observe()
        
        while not terminal:
            #choose best action that is predicted by the network.
            action = model.actions[np.argmax(model.q(state))]
            env.act(action)

            #observe environment
            next_state, reward, terminal = env.observe()

            state = next_state

            if reward == 1:
                win_num+=1
            elif reward == -1:
                lose_num+=1
                
    print("wining rate:{} win:{} lose{}".format(win_num/(win_num+lose_num), win_num, lose_num))