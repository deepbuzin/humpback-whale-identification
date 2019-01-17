from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def euclidean_distance(embeddings):
    prod = tf.matmul(embeddings, tf.transpose(embeddings))
    sq_norms = tf.diag_part(prod)

    dist = tf.expand_dims(sq_norms, 0) - 2.0*prod + tf.expand_dims(sq_norms, 1)
    dist = tf.maximum(dist, 0.0)

    zeros_mask = tf.to_float(tf.equal(dist, 0.0))
    dist = tf.sqrt(dist + zeros_mask*1e-16)
    dist = dist * (1.0-zeros_mask)
    return dist


def valid_triplets_mask(labels):
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
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    mask = tf.logical_and(indices_not_equal, labels_equal)
    return mask


def valid_anchor_negative_mask(labels):
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    mask = tf.logical_not(labels_equal)
    return mask


def batch_all(labels, embeddings, margin, metric=euclidean_distance):
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


def triplet_loss_batch_hard(margin=0.2, metric=euclidean_distance):
    def batch_hard(labels, embeddings):
        dist = metric(embeddings)

        ap_mask = tf.to_float(valid_anchor_positive_mask(labels))
        ap_dist = tf.multiply(dist, ap_mask)
        hardest_positive_dist = tf.reduce_max(ap_dist, axis=1, keepdims=True)

        an_mask = tf.to_float(valid_anchor_negative_mask(labels))
        an_dist = dist + tf.reduce_max(dist, axis=1, keepdims=True) * (1.0-an_mask)
        hardest_negative_dist = tf.reduce_min(an_dist, axis=1, keepdims=True)

        loss = tf.reduce_mean(tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0))
        return loss
    return batch_hard

