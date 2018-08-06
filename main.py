import tensorflow as tf
import argparse
import sys
from model import Dqn
import train
import emulator
import gym
import save_model
import test
import time
from send_mail import send_message

def main(_):
    #make environment of simple video game.
    env = emulator.Emulator()
    #env = gym.make('Breakout-v0')

    sess = tf.Session()

    #build network.
    q_network = Dqn(sess=sess, env=env, name="main")
    target_network = Dqn(sess=sess, env=env, name="target")

    sess.run(tf.global_variables_initializer())

    #initialize all variables, especially copy q_net weights to the target network
    copy_weights_op = train.copy_weights(q_scope_name="main", target_scope_name="target")
    sess.run(copy_weights_op)

    saver = tf.train.Saver()

    if FLAGS.model_dir != '':

        saver.restore(sess, FLAGS.model_dir + "model.ckpt")
        
        model_dir = FLAGS.model_dir
        
    else :
        print("TRAINING")
        start = time.time()
        
        train.train(env=env,
                    model=q_network,
                    target_net=target_network,
                    FLAGS=FLAGS)
        
        print_time(time.time() - start)
        
        send_message()
        
        # if not FLAGS.no_save:
        #     print("The model was not saved.")
        # else :
        model_dir = save_model.save(saver=saver, sess=q_network.sess)
            

    #test 100 episodes.
    #test.simple_test(q_network, 100)



def minitest(env):
    """
    mini test for the simple game
    """
    for i in range(20):
        env.act(1)
        screen, reward, terminal = env.observe()
        print(screen)

        if env.terminal:
            print("reset")
            env.reset()

def print_time(second_time):
    hours, rem = divmod(second_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default='1000', help='Number of training.')
    parser.add_argument('--model_dir', type=str, default='',
                        help='path to model.ckpt file.')
    parser.add_argument('--no_save', action='store_true',
                        help='save sflag.')
    parser.add_argument('--target_update_num', type=int, default='5', help='how often update the target network.')
    parser.add_argument('--minibatch_size', type=int, default='32', help='minibatch size.')
    parser.add_argument('--replay_memory_size', type=int, default='1000', help='replay memory size.')
    parser.add_argument('--gamma', type=float, default='0.9', help='discount factor')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)