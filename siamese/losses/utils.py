from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from utils.metrics import l2_normalized
from utils.curried_functions import tf_equal, tf_multiply, tf_cast
from pyramda import compose

def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.
    Args:
        data: 2-D float `Tensor` of size [n, m].
        mask: 2-D Boolean `Tensor` of size [n, m].
        dim: The dimension over which to compute the maximum.
    Returns:
        masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
    masked_maximums = math_ops.reduce_max(
        math_ops.multiply(data - axis_minimums, mask), dim,

        keepdims=True) + axis_minimums
    return masked_maximums

def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.
    Args:
        data: 2-D float `Tensor` of size [n, m].
        mask: 2-D Boolean `Tensor` of size [n, m].
        dim: The dimension over which to compute the minimum.
    Returns:
        masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
    masked_minimums = math_ops.reduce_min(
        math_ops.multiply(data - axis_maximums, mask), dim,
        keepdims=True) + axis_maximums
    return masked_minimums

def mean_distances(embeddings, labels, metric):
    pdist_matrix = metric(l2_normalized(embeddings))
    adjacency = compose(
        tf_equal(labels),
        array_ops.transpose,
    )(labels)
    
    adjacency_not = compose(
        tf_cast(dtype=dtypes.float32),
        math_ops.logical_not,
    )(adjacency)

    adjacency = math_ops.cast(adjacency, dtype=dtypes.float32)

    positive_mean_distance = compose(
        math_ops.reduce_mean,
        tf_multiply(adjacency),
    )(pdist_matrix)

    negative_mean_distance = compose(
        math_ops.reduce_mean,
        tf_multiply(adjacency_not),
    )(pdist_matrix)

    return positive_mean_distance, negative_mean_distance
