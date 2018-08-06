import tensorflow as tf
import numpy as np
import random
from ops import transform_rgb2gray
from tensorflow.examples.tutorials.mnist import input_data
from collections import deque
from skimage.transform import resize



def train(env, model, target_net, FLAGS):

    #history_length = 4
    replay_memory = deque(maxlen=FLAGS.replay_memory_size)

    for i in range(FLAGS.epoch):
        #initialization of the environment
        env.reset()
        
        # do nothing for random time.
        #for _ in range(random.randint(0, 29)):
        #    screen, _, _, _ = env.step(0)

        #history = [transform_rgb2gray(screen) for _ in range(history_length)]
        #screen = np.stack(history) #[array(image),array(image),...] ==> array([image,image,...])
        step=0
        terminal = False
        screen = env.draw()
        while not terminal:
            #the argent act (epsilon-greedy method.)
            action = epsilon_greedy(screen, model)
            
            #take action and observe the environment
            next_screen, reward, terminal, info = env.step(action)
                
            #store experience to the replay memory.
            if len(replay_memory) >= FLAGS.replay_memory_size:
                replay_memory.popleft()
                
            replay_memory.append((screen, action, reward, next_screen, terminal))
            
            #experience replay
            input_minibatch, target_minibatch = experience_replay(q_net=model,
                                                                    target_net=target_net,
                                                                    replay_memory=replay_memory,
                                                                    FLAGS=FLAGS)

            #learning
            model.sess.run(model.train_op,feed_dict={model.input: input_minibatch,
                                                     model.target: target_minibatch})

            screen = next_screen

            if step % FLAGS.target_update_num == 0:
                #copy q_net weights to the target network.
                op = copy_weights(q_scope_name="main", target_scope_name="target")
                model.sess.run(op)
            
            if reward == 1:
                print("Win")
            elif reward == -1:
                print("Lose")

            step+=1

        
def epsilon_greedy(state, model):
    if np.random.rand() <= model.epsilon:
        #exploration
        return np.random.choice(model.actions)
    else :
        #chose best action that is predicted by the network.
        return model.actions[np.argmax(model.q(state))]
    
def copy_weights(q_scope_name, target_scope_name):
    """
    copy q netwrok weights to  the target netwrok.
    """
    q_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope_name)
    target_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_scope_name)
    
    #operations = [t_var.assign(q_var.value()) for q_var, t_var in zip(q_variables, target_variables)]
    
    operations = []
    
    for q_var, t_var in zip(q_variables, target_variables):
            operations.append(t_var.assign(q_var.value()))
            
    return operations

def experience_replay(q_net, target_net, replay_memory, FLAGS):
    """
    Experience replay algorithm.
    Sample random minibatch of transition from replay memory.

    Set target = reward (for terminal phi)
                    reward + gamma*max_a' Q(phi,a'; sita) (for non-terminal phi)

    Perform a gradient descent step on (y - Q(phi,a; sita))^2
    """

    minibatch_size = min(len(replay_memory), FLAGS.minibatch_size)

    # sample random minibatch of transitions from replay memory.
    #random_batch = np.random.randint(low=0, high=len(replay_memory), size=FLAGS.minibatch_size)
    input_minibatch = []
    target_minibatch = []

    #for idx in random_batch:
    for sample in random.sample(replay_memory, minibatch_size):
        state, action, reward, next_state, terminal = sample

        target = target_net.q(state)

        action_idx = q_net.actions.index(action)

        # update q value target by memory data and itself.(Playing Atari with DRL.p5)
        # Set y_j = r_j      for terminal phi
        #            r_j  + gamma * max_a' Q(phi_j+1, a'; sita)  for non-terminal phi
        if terminal:
            target[action_idx] = reward
        else:
            target[action_idx] = reward + FLAGS.gamma * np.max(target_net.q(next_state))

        input_minibatch.append(state)
        target_minibatch.append(target)

    # train
    # results = self.sess.run([self.train_op, self.loss],
    #                         feed_dict={self.input: input_minibatch, self.target: target_minibatch})

    # self.sess.run(self.loss,  feed_dict={self.input: input_minibatch, self.target: target_minibatch})

    return input_minibatch, target_minibatch


