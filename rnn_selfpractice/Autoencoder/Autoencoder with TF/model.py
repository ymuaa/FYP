import tensorflow as tf
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense

class Seq2SeqModel(object):
    def __init__(self, num_units_per_cell, num_layers_per_cell, encoder_vocab_size, decoder_vocab_size, embedding_dim,
                 grad_clip, is_inference=False):
        # input
        self.input_x = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')

        # embedding
        with tf.variable_scope('embedding'):
            encoder_embedding = tf.Variable(tf.truncated_normal(shape=[encoder_vocab_size, embedding_dim], stddev=0.1),
                                            name='encoder_embedding')
            decoder_embedding = tf.Variable(tf.truncated_normal(shape=[decoder_vocab_size, embedding_dim], stddev=0.1),
                                            name='decoder_embedding')
        # encoder
        with tf.variable_scope('encoder'):
            encoder = self._get_simple_lstm(num_units_per_cell, num_layers_per_cell)

        with tf.device('/cpu:0'):
            input_x_embedded = tf.nn.embedding_lookup(encoder_embedding, self.input_x)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder, input_x_embedded, dtype=tf.float32)

        print 'shape of encoder_state: ', encoder_state[0]

        if not is_inference:
            self.target_ids = tf.placeholder(tf.int32, shape=[None, None], name='target_ids')
            self.decoder_seq_length = tf.placeholder(tf.int32, shape=[None], name='batch_seq_length')
            with tf.device('/cpu:0'):
                target_embeddeds = tf.nn.embedding_lookup(decoder_embedding, self.target_ids)
                # convert target id into 'embedded id' (then later, link the 'embedded target id' and 'embedded input id') together
            helper = TrainingHelper(target_embeddeds, self.decoder_seq_length)
        else:
            self.start_tokens = tf.placeholder(tf.int32, shape=[None], name='start_tokens')
            self.end_token = tf.placeholder(tf.int32, name='end_token')
            helper = GreedyEmbeddingHelper(decoder_embedding, self.start_tokens, self.end_token)

        with tf.variable_scope('decoder'):
            fc_layer = Dense(units=decoder_vocab_size) # units: dimensionality of the output space
            decoder_cell = self._get_simple_lstm(num_units_per_cell, num_layers_per_cell)
            decoder = BasicDecoder(decoder_cell, helper, encoder_state, fc_layer)

        logits, final_state, final_sequence_length = dynamic_decode(decoder)
        # logtis is composed of (1) rnn_output (2) sample_id

        if not is_inference:
            targets = tf.reshape(self.target_ids, [-1])
            logits_flat = tf.reshape(logits.rnn_output, [-1, decoder_vocab_size])

            print 'shape of target_ids: ', self.target_ids.shape
            print 'shape of logtis: ', logits.rnn_output.shape
            print 'shape of logtis_flat:{}'.format(logits_flat.shape)

            self.cost = tf.losses.sparse_softmax_cross_entropy(targets, logits_flat)
            print 'shape of cost: ', self.cost.shape

            # define train op
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)

            optimizer = tf.train.AdamOptimizer(1e-3)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        else:
            self.prob = tf.nn.softmax(logits)

    def train(self, num_epochs, num_steps = 200, batch_size = 32):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            training_losses = []

            for epoch in range(num_epochs):
                pass



    def _get_simple_lstm(self, num_units, num_layers):
        lstm_layers = [tf.contrib.rnn.LSTMCell(num_units) for _ in xrange(num_layers)]
        return tf.contrib.rnn.MultiRNNCell(lstm_layers)


if __name__ == '__main__':
    model = Seq2SeqModel(num_units_per_cell=10, num_layers_per_cell=1, encoder_vocab_size=20, decoder_vocab_size=20, embedding_dim=20,
                 grad_clip=0.5)
