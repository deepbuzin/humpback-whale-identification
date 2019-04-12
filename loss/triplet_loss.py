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


def euclidean_dist(embeddings):
    prod = tf.matmul(embeddings, tf.transpose(embeddings))
    #sq_norms = tf.reduce_sum(tf.square(embeddings), axis=1)
    sq_norms = tf.diag_part(prod)
    dist = tf.reshape(sq_norms, (-1, 1)) - 2 * prod + tf.reshape(sq_norms, (1, -1))
    return dist

def soft_margin_triplet_loss_(labels, embeddings):
    inf = tf.constant(1e+9, tf.float32)
    epsilon = tf.constant(1e-6, tf.float32)
    zero = tf.constant(0, tf.float32)

    dist = tf.sqrt(tf.maximum(zero, epsilon + euclidean_dist(embeddings)))
    # mask matrix showing equal labels of embeddings
    equal_label_mask = tf.cast(tf.equal(tf.reshape(labels, (-1, 1)), tf.reshape(labels, (1, -1))), tf.float32)

    pos_dist = tf.reduce_max(equal_label_mask * dist, axis=1)
    neg_dist = tf.reduce_min((equal_label_mask * inf) + dist, axis=1)

    loss = tf.reduce_mean(tf.nn.softplus(pos_dist - neg_dist))
    return loss


def soft_margin_triplet_loss(labels, embeddings):
    #embeddings -= tf.reduce_mean(embeddings, axis=0)  # !!!!!!!!!!!!!!!!!!

    inf = tf.constant(1e+9, tf.float32)
    epsilon = tf.constant(1e-6, tf.float32)
    zero = tf.constant(0, tf.float32)

    #dist = tf.sqrt(tf.maximum(zero, epsilon + euclidean_dist(embeddings)))
    dist = tf.maximum(zero, epsilon + euclidean_dist(embeddings))  # sqeuclidean
    # mask matrix showing equal labels of embeddings
    equal_label_mask = tf.cast(tf.equal(tf.reshape(labels, (-1, 1)), tf.reshape(labels, (1, -1))), tf.float32)

    pos_dist = tf.reduce_max(equal_label_mask * dist, axis=1)
    neg_dist = tf.reduce_min((equal_label_mask * inf) + dist, axis=1)

    loss = tf.reduce_mean(tf.nn.softplus(pos_dist - neg_dist))

    #hard loss
    #margin = tf.constant(1.5, tf.float32)
    #loss = tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))
    return loss

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

def pairwise_distance(feature, squared=False):
  """Computes the pairwise distance matrix with numerical stability.
  output[i, j] = || feature[i, :] - feature[j, :] ||_2
  Args:
    feature: 2-D Tensor of size [number of data, feature dimension].
    squared: Boolean, whether or not to square the pairwise distances.
  Returns:
    pairwise_distances: 2-D Tensor of size [number of data, number of data].
  """
  pairwise_distances_squared = math_ops.add(
      math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),
      math_ops.reduce_sum(
          math_ops.square(array_ops.transpose(feature)),
          axis=[0],
          keepdims=True)) - 2.0 * math_ops.matmul(feature,
                                                  array_ops.transpose(feature))

  # Deal with numerical inaccuracies. Set small negatives to zero.
  pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
  # Get the mask where the zero distances are at.
  error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

  # Optionally take the sqrt.
  if squared:
    pairwise_distances = pairwise_distances_squared
  else:
    pairwise_distances = math_ops.sqrt(
        pairwise_distances_squared +
        math_ops.cast(error_mask, dtypes.float32) * 1e-16)

  # Undo conditionally adding 1e-16.
  pairwise_distances = math_ops.multiply(
      pairwise_distances,
      math_ops.cast(math_ops.logical_not(error_mask), dtypes.float32))

  num_data = array_ops.shape(feature)[0]
  # Explicitly set diagonals to zero.
  mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
      array_ops.ones([num_data]))
  pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
  return pairwise_distances

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

def tf_contrib_loss(margin=1.0):
    def tf_contrib_hard_loss(labels, embeddings):
      """Computes the triplet loss with semi-hard negative mining.
      The loss encourages the positive distances (between a pair of embeddings with
      the same labels) to be smaller than the minimum negative distance among
      which are at least greater than the positive distance plus the margin constant
      (called semi-hard negative) in the mini-batch. If no such negative exists,
      uses the largest negative distance instead.
      See: https://arxiv.org/abs/1503.03832.
      Args:
        labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
          multiclass integer labels.
        embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
          be l2 normalized.
        margin: Float, margin term in the loss definition.
      Returns:
        triplet_loss: tf.float32 scalar.
      """
      # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
      # lshape = array_ops.shape(labels)
      # assert lshape.shape == 1
      # labels = array_ops.reshape(labels, [lshape[0], 1])

      labels = tf.reshape(labels, (-1, 1))

      # Build pairwise squared distance matrix.
      pdist_matrix = pairwise_distance(embeddings, squared=True)
      # Build pairwise binary adjacency matrix.
      adjacency = math_ops.equal(labels, array_ops.transpose(labels))
      # Invert so we can select negatives only.
      adjacency_not = math_ops.logical_not(adjacency)

      batch_size = array_ops.size(labels)

      # Compute the mask.
      pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
      mask = math_ops.logical_and(
          array_ops.tile(adjacency_not, [batch_size, 1]),
          math_ops.greater(
              pdist_matrix_tile, array_ops.reshape(
                  array_ops.transpose(pdist_matrix), [-1, 1])))
      mask_final = array_ops.reshape(
          math_ops.greater(
              math_ops.reduce_sum(
                  math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True),
              0.0), [batch_size, batch_size])
      mask_final = array_ops.transpose(mask_final)

      adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
      mask = math_ops.cast(mask, dtype=dtypes.float32)

      # negatives_outside: smallest D_an where D_an > D_ap.
      negatives_outside = array_ops.reshape(
          masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
      negatives_outside = array_ops.transpose(negatives_outside)

      # negatives_inside: largest D_an.
      negatives_inside = array_ops.tile(
          masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
      semi_hard_negatives = array_ops.where(
          mask_final, negatives_outside, negatives_inside)

      loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

      mask_positives = math_ops.cast(
          adjacency, dtype=dtypes.float32) - array_ops.diag(
              array_ops.ones([batch_size]))

      # In lifted-struct, the authors multiply 0.5 for upper triangular
      #   in semihard, they take all positive pairs except the diagonal.
      num_positives = math_ops.reduce_sum(mask_positives)

      triplet_loss = math_ops.truediv(
          math_ops.reduce_sum(
              math_ops.maximum(
                  math_ops.multiply(loss_mat, mask_positives), 0.0)),
          num_positives,
          name='triplet_semihard_loss')

      return triplet_loss

    return tf_contrib_hard_loss
