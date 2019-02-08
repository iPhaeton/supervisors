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
        log_every = kwargs.pop('log_every', 5)
        session = kwargs.get('session')
        observer = kwargs.get('observer', None)

        shutil.rmtree(log_dir, ignore_errors=True)

        if log_dir != None:
            writer = tf.summary.FileWriter(log_dir)
            writer.add_graph(session.graph)
            merged_summary = tf.summary.merge_all()

            def log_summary(i, feed_dict):
                if i % log_every != 0:
                    return

                s = session.run(merged_summary, feed_dict)
                writer.add_summary(s, i)

            if observer != None:
                observer.add_listener(ON_ITER_START, log_summary)
        
        return func(**kwargs)
    
    return wrapper