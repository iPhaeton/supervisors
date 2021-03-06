import tensorflow as tf
from pyramda import curry
from utils.common import filter_list

tf_cast = curry(tf.cast)
tf_add = curry(tf.add)
tf_multiply = curry(tf.multiply)
tf_equal = curry(tf.equal)

filter_list = curry(filter_list)