from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def euclidean_distance(embeddings):
    """Compute the 2D matrix of pairwise euclidean distances between embeddings.

    :param embeddings: tensor of shape (batch_size, embedding_size)
    :return dist: tensor of shape (batch_size, batch_size)
    """
    prod = tf.matmul(embeddings, tf.transpose(embeddings))
    sq_norms = tf.diag_part(prod)

    dist = tf.expand_dims(sq_norms, 0) - 2.0*prod + tf.expand_dims(sq_norms, 1)
    dist = tf.maximum(dist, 0.0)

    zeros_mask = tf.to_float(tf.equal(dist, 0.0))
    dist = tf.sqrt(dist + zeros_mask*1e-16)
    dist = dist * (1.0-zeros_mask)
    return dist


def valid_triplets_mask(labels):
    """Compute the 3D boolean mask where mask[a, p, n] is True if (a, p, n) is a valid triplet,
    as in a, p, n are distinct and labels[a] == labels[p], labels[a] != labels[n].

    :param labels: tensor of shape (batch_size,)
    :return mask: tf.bool tensor of shape (batch_size, batch_size, batch_size)
    """
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)
    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)
    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    mask = tf.logical_and(distinct_indices, valid_labels)
    return mask


def valid_anchor_positive_mask(labels):
    """Compute a 2D boolean mask where mask[a, p] is True if a and p are distinct and have the same label.

    :param labels: tensor of shape (batch_size,)
    :return mask: tf.bool tensor of shape (batch_size, batch_size)
    """
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    mask = tf.logical_and(indices_not_equal, labels_equal)
    return mask


def valid_anchor_negative_mask(labels):
    """Compute a 2D boolean mask where mask[a, n] is True if a and n have distinct label.

    :param labels: tensor of shape (batch_size,)
    :return mask: tf.bool tensor of shape (batch_size, batch_size)
    """
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    mask = tf.logical_not(labels_equal)
    return mask


def triplet_loss(margin=0.2, strategy='batch_all', metric=euclidean_distance):
    """Compute the triplet loss over the batch of embeddings.

    :param margin: margin that is going to be enforced by the triplet loss
    :param strategy: string, that indicated whether we're using the 'batch hard' or the 'batch all' mining strategy
    :param metric: a callback function that we use to calculate the distance between each pair of vectors
    :return: a callback function that calculates the loss according to the specified mining strategy
    """
    def batch_all(labels, embeddings):
        """Compute the loss by generating all the valid triplets and averaging over the positive ones

        :param labels: tensor of shape (batch_size,)
        :param embeddings: tensor of shape (batch_size, embedding_size)
        :return loss: scalar tensor
        """
        dist = metric(embeddings)

        anchor_positive_dist = tf.expand_dims(dist, 2)
        anchor_negative_dist = tf.expand_dims(dist, 1)

        loss = anchor_positive_dist - anchor_negative_dist + margin

        mask = tf.to_float(valid_triplets_mask(labels))
        loss = tf.multiply(loss, mask)
        loss = tf.maximum(loss, 0.0)

        num_non_easy_triplets = tf.reduce_sum(tf.to_float(tf.greater(loss, 1e-16)))
        loss = tf.reduce_sum(loss) / (num_non_easy_triplets + 1e-16)
        return loss

    def batch_hard(labels, embeddings):
        """Compute the loss on the triplet consisting of the hardest positive and the hardest negative

        :param labels: tensor of shape (batch_size,)
        :param embeddings: tensor of shape (batch_size, embedding_size)
        :return loss: scalar tensor
        """
        dist = metric(embeddings)

        ap_mask = tf.to_float(valid_anchor_positive_mask(labels))
        ap_dist = tf.multiply(dist, ap_mask)
        hardest_positive_dist = tf.reduce_max(ap_dist, axis=1, keepdims=True)

        an_mask = tf.to_float(valid_anchor_negative_mask(labels))
        an_dist = dist + tf.reduce_max(dist, axis=1, keepdims=True) * (1.0-an_mask)
        hardest_negative_dist = tf.reduce_min(an_dist, axis=1, keepdims=True)

        loss = tf.reduce_mean(tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0))
        return loss

    if strategy == 'batch_all':
        return batch_all
    elif strategy == 'batch_hard':
        return batch_hard
    else:
        raise ValueError('unknown strategy: %s' % strategy)

