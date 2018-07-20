import tensorflow as tf
from DataHelper import random_sequences, make_batch

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

PAD = 0
EOS = 1

class Seq2Seq_1():
    def __init__(self, vocab_size=10, embedding_size=20, encoder_num_units=20, decoder_num_units=20):
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            print "ENTERED!!! "
            self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_input')
            self.decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_input')
            self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

            # encoder and decoder share same embedding
            self.embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), dtype=tf.float32)
            encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
            decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)

            encoder_cell = tf.contrib.rnn.LSTMCell(encoder_num_units)
            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                encoder_cell, encoder_inputs_embedded, dtype=tf.float32, time_major=True
            )

            decoder_cell = tf.contrib.rnn.LSTMCell(decoder_num_units)
            decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
                decoder_cell, decoder_inputs_embedded, dtype=tf.float32, time_major=True, scope='plain_decoder',
                initial_state=encoder_final_state
            )

            decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)
            self.decoder_prediction = tf.argmax(decoder_logits, 2)

            stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(self.decoder_targets, depth=vocab_size, dtype=tf.float32),
                logits=decoder_logits
            )
            self.loss = tf.reduce_mean(stepwise_cross_entropy)
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

            # print "shape of decoder_inputs_embedded: ", decoder_inputs_embedded.shape
            # print "shape of decoder_final_state:", (decoder_final_state)
            # print "shape of decoder_outputs: ", decoder_outputs.shape
            # print "shape of decoder_logits: ", decoder_logits.shape
            # print "shape of decoder_prediction: ", self.decoder_prediction.shape
            # print "shape of decoder_targets: ", self.decoder_targets.shape
            # print "shape of one-hot decoder_targets: ", tf.one_hot(self.decoder_targets, depth=vocab_size, dtype=tf.float32).shape
            # print "shape of stepwise_cross_entropy: ", stepwise_cross_entropy.shape
            # print "shape of self.loss: ", self.loss

    def train(self, batches, num_epochs=3001):
        loss_track = []

        with tf.Session(graph=self.train_graph) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(num_epochs):
                batch = next(batches)
                encoder_inputs_, _ = make_batch(batch)
                decoder_targets_, _ = make_batch([(sequence) + [EOS] for sequence in batch])
                decoder_inputs_, _ = make_batch([[EOS] + (sequence) for sequence in batch])

                feed_dict = {self.encoder_inputs: encoder_inputs_,
                             self.decoder_inputs: decoder_inputs_,
                             self.decoder_targets: decoder_targets_}

                _, l = sess.run([self.train_op, self.loss], feed_dict)
                loss_track.append(l)
                if epoch == 0 or epoch % 1000 == 0:
                    print 'loss: {}'.format(sess.run(self.loss, feed_dict))
                    predict_ = sess.run(self.decoder_prediction, feed_dict)
                    for i, (inp, pred) in enumerate(zip(feed_dict[self.encoder_inputs].T, predict_.T)):
                        print('input > {}'.format(inp))
                        print('predicted > {}'.format(pred))
                        if i >= 10:
                            break

        plt.plot(loss_track)
        plt.show()

if __name__ == '__main__':
    batch_size = 100
    batches = random_sequences(length_min=3, length_max=10, vocab_min=2, vocab_max=10, batch_size=batch_size)

    vocab_size = 10
    embedding_size = 20
    encoder_hidden_units = 20
    decoder_hidden_units = 20
    batch_size = 100

    model = Seq2Seq_1(vocab_size=vocab_size, embedding_size=embedding_size, encoder_num_units=encoder_hidden_units,
                      decoder_num_units=decoder_hidden_units)
    model.train(batches=batches)