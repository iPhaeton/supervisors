import tensorflow as tf
import shutil
from constants import ON_ITER_START, ON_ITER_END

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
        observer = kwargs.get('observer', None)
        
        if observer != None:
            observer.add_listener(ON_ITER_START, lambda x: print('Iter started', x))
            observer.add_listener(ON_ITER_END, lambda x: print('Iter ended', x))

        shutil.rmtree(log_dir, ignore_errors=True)

        if log_dir != None:
            writer = tf.summary.FileWriter(log_dir)
            writer.add_graph(session.graph)
        
        return func(**kwargs)
    
    return wrapper