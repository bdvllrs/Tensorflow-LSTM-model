import tensorflow as tf


class LSTM:
    """The LSTM Model"""

    def __init__(self, batch_size, embedding_size, vocab_size, hidden_size, max_size, teacher_forcing=True):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_size = max_size
        self.teacher_forcing = teacher_forcing

    def __call__(self, x, label=None):
        assert self.teacher_forcing and label is not None, "Please provide the label to use teacher forcing."

        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)

        # We give in the vector [bos, tag1, tag2, ..., last tag]
        # and expect to receive [tag1, tag2, ..., sos]
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            word_embeddings = tf.get_variable("word_embeddings",
                                              [self.vocab_size, self.embedding_size], dtype=tf.int32)

        inputs = tf.nn.embedding_lookup(word_embeddings, x)
        if label is not None:
            embedded_labels = tf.nn.embedding_lookup(word_embeddings, label)

        # Sets the size into time x batch x embedding
        x_t = tf.transpose(inputs, [1, 0, 2], name="x_embedded_transposed")

        if label is not None:
            label_t = tf.transpose(embedded_labels, [1, 0, 2], name="y_embedded_transposed")

        # weight for the softmax
        with tf.variable_scope("softmax", reuse=tf.AUTO_REUSE):
            W = tf.get_variable("W", shape=(self.max_size - 1, self.batch_size,
                                            self.vocab_size, self.hidden_size),
                                initializer=tf.contrib.layers.xavier_initializer())

        default_state = tf.nn.rnn_cell.LSTMStateTuple(tf.zeros([self.batch_size, self.hidden_size],
                                                               name="state1"),
                                                      tf.zeros([self.batch_size, self.hidden_size],
                                                               name="state2"))

        def body(init, vect):
            state, k = init
            # Get the word of time step k
            # Teacher forcing
            if not self.teacher_forcing or k == 0:
                in_vect = vect  # put the last generated vector in
            else:
                in_vect = label_t[k - 1]  # Put the last label in for faster training
            in_vect = tf.cast(in_vect, tf.float32)
            _, state = rnn_cell(in_vect, state)
            return (state, k + 1)

        # Counter to count the number of words (max max_size)
        k = 0

        state, _ = tf.scan(body, x_t, (default_state, k))
        output = state.h
        print(state)
        # Add a dimension for being able to multiply with W
        final_output = tf.expand_dims(output, 3)
        final_output = tf.matmul(W, final_output)  # Premier
        print(final_output.shape)
        final_output = tf.squeeze(final_output, 3)
        final_output = tf.transpose(final_output, [1, 0, 2])
        return final_output, tf.nn.softmax(final_output)



    def optimize(self,output, label, learning_rate):
        training_vars = tf.trainable_variables()
        # En fait je ne suis pas sur, j'ai l'impression que la fonction sparse_softmax_cross_entropy_with_logits calcul un softmax
        # vu son nom. En plus il prend les `logits`... donc effectivement, il ne faut pas faire de softmax avant...
        # Le problème c'est pour l'évaluation... Vu que cette fonction n'est utilisée que pour l'entraînement.
        # Il faudrait donc vérifier en fonction de si `label is None` ou pas pour savoir si on entraîne ou si on
        # évalu... Si on évalu, il faudrait garde le premier softmax
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=output))  # Deuxième...
        self.loss=tf.reduce_mean(cross_entropy)
        grads, _ = tf.clip_by_global_norm(tf.gradients(cross_entropy, training_vars), 5)  # Max gradient of 5

        optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer.apply_gradients(zip(grads, training_vars))

        return optimizer.minimize(cross_entropy)
