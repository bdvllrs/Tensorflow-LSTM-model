import tensorflow as tf


def lstm(x, label, vocab_size, hidden_size, max_size, batch_size, embedding_size, teacher_forcing, down_project=False):
    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)

    # We give in the vector [bos, tag1, tag2, ..., last tag]
    # and expect to receive [tag1, tag2, ..., sos]
    with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
        word_embeddings = tf.get_variable("word_embeddings",
                                          [vocab_size, embedding_size], dtype=tf.float32)

    inputs = tf.nn.embedding_lookup(word_embeddings, x)
    if label is not None:
        embedded_labels = tf.nn.embedding_lookup(word_embeddings, label)
        label_t = tf.transpose(embedded_labels, [1, 0, 2], name="y_embedded_transposed")

    # Sets the size into time x batch_size x embedding
    x_t = tf.transpose(inputs, [1, 0, 2], name="x_embedded_transposed")
    W, WP = None, None
    # weight for the softmax
    with tf.variable_scope("softmax", reuse=tf.AUTO_REUSE):
        if down_project:
            #instead of a lot of weight vectors per softmax output, less weight vectors per softmax output.
            #remember, softmax layer has fixed size. But for each node, we can influence how much connections (w) enter
            #so down project is just simply a additional dense layer between state and final out
            
            #downProject * vocab
            W = tf.get_variable("W", shape=(down_project, vocab_size),
                                initializer=tf.contrib.layers.xavier_initializer())
            #hidden * downProject
            WP = tf.get_variable("WP", shape=(hidden_size, down_project),
                                 initializer=tf.contrib.layers.xavier_initializer())
        else:
            #hidden * vocab
            W = tf.get_variable("W", shape=(hidden_size, vocab_size),
                                initializer=tf.contrib.layers.xavier_initializer())

    #2*[batch_size * hidden]
    default_state = tf.nn.rnn_cell.LSTMStateTuple(tf.zeros([batch_size, hidden_size],
                                                           name="state1"),
                                                  tf.zeros([batch_size, hidden_size],
                                                           name="state2"))
    #previous output, current input
    def body(init, vect):
        #(state, hidden) and k
        state, k = init
        # Get the word of time step k
        # Teacher forcing
        # is_teacher_forcing = not tf.less(teacher_forcing, tf.constant(1, dtype=tf.int8))
        in_vect = tf.cond(tf.logical_and(teacher_forcing, k != 0), lambda: label_t[k - 1], lambda: vect)
        # if not is_teacher_forcing or k == 0:
        #     in_vect = vect  # put the last generated vector in
        # else:
        #     in_vect = label_t[k - 1]  # Put the last label in for faster training
        in_vect = tf.cast(in_vect, tf.float32)
        #run on the rnn, return output and new states (a tuple of hidden state c and output h)
        _, state = rnn_cell(in_vect, state)
        return state, k + 1

    # Counter to count the number of words (max max_size)
    k = 0

    #tf.scan: function,elements,initializer (=init state, timstep now = 0)
    #tf.scan over elements 1st dimension: max_size*batch_size*vocab
    #funciton: previous, current
    #init: zero states, counter = 0
    state, _ = tf.scan(body, x_t, (default_state, k)) # 2*[max_size , bs , hidden]
    output = state.h #[max_size * bs * hs] getting output of lstmCell which is denoted by h
    #we split the state to be able to access only the last timestep
    #go into tuple
    states_split_cellstate = tf.split(state[0], max_size-1, axis = 0)
    states_split_hiddenstate = tf.split(state[1], max_size-1, axis = 0)
    lastState = tf.nn.rnn_cell.LSTMStateTuple(tf.squeeze(states_split_cellstate[-1]),tf.squeeze(states_split_hiddenstate[-1]))
    allState = state
    # Add a dimension for being able to multiply with W
    # final_output = tf.expand_dims(output, 3)
    final_output = tf.reshape(output, [(max_size - 1) * batch_size, -1]) #[(max_size * batch_size) , hidden]
    if down_project:
        final_output = tf.matmul(final_output, WP)
    final_output = tf.matmul(final_output, W)  # Premier
    final_output = tf.reshape(final_output, [max_size - 1, batch_size, -1])
    final_output = tf.transpose(final_output, [1, 0, 2])
    return word_embeddings, final_output, tf.nn.softmax(final_output, name="softmax_output"), allState, lastState, rnn_cell, W, WP


def optimize(output, label, learning_rate):
    training_vars = tf.trainable_variables()
    # Weights to get rid of the loss for <pad> token
    # We are assuming here that the token is 0...
    weights = tf.cast(tf.greater(label, 0), tf.float32, name="weights")
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=output)
    cross_entropy_batch = tf.reduce_sum(weights * cross_entropy, axis=1) / tf.reduce_sum(weights, axis=1)  # Deuxième...
    grads, _ = tf.clip_by_global_norm(tf.gradients(cross_entropy, training_vars), 5)  # Max gradient of 5
    optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer.apply_gradients(zip(grads, training_vars))
    opt = optimizer.minimize(cross_entropy_batch, name="optimizer")
    cross_entropy = tf.identity(cross_entropy_batch, name="cross_entropy")  # for sentences
    cross_entropy_batch = tf.reduce_mean(cross_entropy_batch, name="cross_entropy_batch")  # For the whole batch

    return opt, cross_entropy_batch, cross_entropy, weights



def getOneStep(down_project, hidden_size, rnn_cell, W, WP=None):
    #given the last step's state from the full sized rnn, we provide a new step and only unroll one step
    #1, batch_size, alphsize
    lastInput_tensor = tf.placeholder(tf.int32, [1, 1], name='lastInput')
    lastState_a_tensor = tf.placeholder(tf.float32, [1, hidden_size], name='lastState_a')
    lastState_b_tensor = tf.placeholder(tf.float32, [1, hidden_size], name='lastState_b')
    lastState = tf.nn.rnn_cell.LSTMStateTuple(lastState_a_tensor,lastState_b_tensor)
    # rnn_cell_pred = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    rnn_cell_pred = rnn_cell
    lastInput_tensor = tf.cast(lastInput_tensor, tf.float32)
    _, lastState = rnn_cell_pred(lastInput_tensor,lastState)
    output = lastState.h
    final_output = output
    if down_project:
        with tf.variable_scope("softmax", reuse=tf.AUTO_REUSE):
            final_output = tf.matmul(final_output, WP)
    final_output = tf.matmul(final_output, W)  # Premier
    final_output = tf.expand_dims(final_output, 1)
    final_output = tf.transpose(final_output, [1, 0, 2])
    return lastInput_tensor, final_output, tf.nn.softmax(final_output), lastState_a_tensor, lastState_b_tensor

def getDefaultState(batch_size,hidden_size):
    default_state = tf.nn.rnn_cell.LSTMStateTuple(tf.zeros([batch_size, hidden_size]),tf.zeros([batch_size, hidden_size]))
    return default_state
