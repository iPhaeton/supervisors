import tensorflow as tf
import shutil

def partially_applied(func):
    def outer_wrapper(**kwargs):
        def inner_wrapper(*args):
            return func(*args, **kwargs)
        return inner_wrapper
    return outer_wrapper

def with_tensorboard(func):
    def wrapper(**kwargs):
        log_dir = kwargs.pop('log_dir', None)
        session = kwargs.get('session')

        shutil.rmtree(log_dir, ignore_errors=True)

        if log_dir != None:
            writer = tf.summary.FileWriter(log_dir)
            writer.add_graph(session.graph)
        
        return func(**kwargs)
    
    return wrapper