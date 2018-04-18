import tensorflow as tf


def lstm(x, label, vocab_size, hidden_size, max_size, batch_size, embedding_size, teacher_forcing):
    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)

    # We give in the vector [bos, tag1, tag2, ..., last tag]
    # and expect to receive [tag1, tag2, ..., sos]
    with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
        word_embeddings = tf.get_variable("word_embeddings",
                                          [vocab_size, embedding_size], dtype=tf.int32)

    inputs = tf.nn.embedding_lookup(word_embeddings, x)
    if label is not None:
        embedded_labels = tf.nn.embedding_lookup(word_embeddings, label)
        label_t = tf.transpose(embedded_labels, [1, 0, 2], name="y_embedded_transposed")

    # Sets the size into time x batch x embedding
    x_t = tf.transpose(inputs, [1, 0, 2], name="x_embedded_transposed")

    # weight for the softmax
    with tf.variable_scope("softmax", reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", shape=(hidden_size, vocab_size),
                            initializer=tf.contrib.layers.xavier_initializer())

    default_state = tf.nn.rnn_cell.LSTMStateTuple(tf.zeros([batch_size, hidden_size],
                                                           name="state1"),
                                                  tf.zeros([batch_size, hidden_size],
                                                           name="state2"))

    def body(init, vect):
        state, k = init
        # Get the word of time step k
        # Teacher forcing
        # is_teacher_forcing = not tf.less(teacher_forcing, tf.constant(1, dtype=tf.int8))
        in_vect = tf.cond(tf.logical_and(teacher_forcing, k != 0), lambda: label_t[k-1], lambda: vect)
        # if not is_teacher_forcing or k == 0:
        #     in_vect = vect  # put the last generated vector in
        # else:
        #     in_vect = label_t[k - 1]  # Put the last label in for faster training
        in_vect = tf.cast(in_vect, tf.float32)
        _, state = rnn_cell(in_vect, state)
        return state, k + 1

    # Counter to count the number of words (max max_size)
    k = 0

    state, _ = tf.scan(body, x_t, (default_state, k))
    output = state.h
    # Add a dimension for being able to multiply with W
    # final_output = tf.expand_dims(output, 3)
    final_output = tf.reshape(output, [(max_size-1) * batch_size, -1])
    final_output = tf.matmul(final_output, W)  # Premier
    final_output = tf.reshape(final_output, [max_size-1, batch_size, -1])
    final_output = tf.transpose(final_output, [1, 0, 2])
    return final_output, tf.nn.softmax(final_output)


def optimize(output, label, learning_rate):
    training_vars = tf.trainable_variables()
    
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=output)
    cross_entropy_batch = tf.reduce_mean(cross_entropy)  # Deuxième...
    grads, _ = tf.clip_by_global_norm(tf.gradients(cross_entropy, training_vars), 5)  # Max gradient of 5
    optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer.apply_gradients(zip(grads, training_vars))

    return optimizer.minimize(cross_entropy), cross_entropy_batch, cross_entropy
