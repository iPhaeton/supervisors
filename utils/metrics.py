import tensorflow as tf

def cosine_distance(embeddings):
    """
    Compute cosine distance matrix
    
    Parameters:
    -----------
    - embeddings: Tensor(N, E)
        Image embeddings, outputs of the convolutional network.
        N - number of samples (None)
        E - embedding size
    """
    
    normalized_embeddings = tf.divide(
        embeddings,
        tf.norm(embeddings),
    )
    
    return tf.subtract(
        1.,
        tf.matmul(normalized_embeddings, tf.transpose(normalized_embeddings))
    )