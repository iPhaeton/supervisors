import tensorflow as tf
import shutil
from constants import ON_ITER_START, ON_ITER_END
import os

def partially_applied(func):
    def outer_wrapper(**kwargs):
        def inner_wrapper(*args):
            return func(*args, **kwargs)
        return inner_wrapper
    return outer_wrapper

def with_tensorboard(func):
    def wrapper(*args, **kwargs):
        log_dir = kwargs.pop('log_dir', None)
        log_every = kwargs.pop('log_every', 5)
        session = kwargs.get('session')
        observer = kwargs.get('observer', None)

        shutil.rmtree(log_dir, ignore_errors=True)

        if log_dir != None:
            writer = tf.summary.FileWriter(log_dir)
            writer.add_graph(session.graph)
            merged_summary = tf.summary.merge_all()

            def log_summary(i, feed_dict, summaries):
                if i % log_every != 0:
                    return
                
                summaries.append(merged_summary)
                calculated_summaries = session.run(summaries, feed_dict)
                for s in calculated_summaries:
                    writer.add_summary(s, i)

            if observer != None:
                observer.add_listener(ON_ITER_START, log_summary)
        
        return func(*args, **kwargs)
    
    return wrapper

def with_saver(func):
    def wrapper(*args, **kwargs):
        saver = tf.train.Saver()
        save_dir = kwargs.pop('save_dir', None)
        save_every = kwargs.pop('save_every', 5)
        session = kwargs.get('session')
        observer = kwargs.get('observer', None)

        def save(i, _):
            if (i % save_every == 0) & (i != 0):
                saver.save(session, os.path.join(save_dir, f'iteration-{i}.ckpt'))

        if observer != None:
            observer.add_listener(ON_ITER_END, save)

        return func(*args, **kwargs)

    return wrapper