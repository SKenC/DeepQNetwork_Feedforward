import datetime 
import os

def save(saver, sess):
    """save model.
    Args:
        saver:    tf.Saver()
        sess:     trained model's session.
    Return:
        dir:  directory path to the model.ckpt
    """
    time = datetime.datetime.today()
    time_str = "%s-%s-%s-%s"%(time.month, 
                           time.day, 
                           time.hour, 
                           time.minute)

    os.makedirs("./model/"+time_str)

    saver.save(sess, "./model/"+time_str+"/model.ckpt")
    
    print("The model saved at ./model/"+time_str+"/model.ckpt")
    
    return "./model/"+time_str+"/"

    