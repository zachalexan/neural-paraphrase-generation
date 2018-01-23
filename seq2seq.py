import tensorflow as tf
from tensorflow.contrib import layers

class Seq2seq:
    def __init__(self, vocab_size, FLAGS):
        self.FLAGS = FLAGS
        self.vocab_size = vocab_size

        # Encoding
    def encode(self, seq, reuse=None):
        # input_lengths  = tf.reduce_sum(tf.to_int32(tf.not_equal(seq, 1)), 1)
        input_embed    = layers.embed_sequence(seq,
                                               vocab_size=self.vocab_size,
                                               embed_dim = self.embed_dim,
                                               scope = 'embed',
                                               reuse = reuse)
        cell = tf.contrib.rnn.LSTMCell(num_units=self.num_units, reuse=reuse)
        # if self.FLAGS.use_residual_lstm:
        #     cell = tf.contrib.rnn.ResidualWrapper(cell)
        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell, input_embed, dtype=tf.float32)
        encoder_final_state_vec = tf.concat(encoder_final_state, 1)
        return encoder_final_state, encoder_final_state_vec
        # return encoder_outputs, encoder_final_state, input_lengths

    def decode(self, encoder_out, scope, output=None, mode='train', reuse=None):

        # From the encoder
        # encoder_outputs = encoder_out[0]
        encoder_state = encoder_out[0]
        # input_lengths = encoder_out[2]

        # Perform the embedding
        if mode=='train':
            if output is None:
                raise Exception('output must be provided for mode=train')
            train_output   = tf.concat([tf.expand_dims(self.start_tokens, 1), output], 1)
            output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(train_output, 1)), 1)
            output_embed   = layers.embed_sequence(
                train_output,
                vocab_size=self.vocab_size,
                embed_dim = self.embed_dim,
                scope = 'encode/embed', reuse = True)

        # Prepare the helper
        if mode=='train':
            helper = tf.contrib.seq2seq.TrainingHelper(output_embed, output_lengths)
        if mode=='predict':
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.embeddings,
                start_tokens=tf.to_int32(self.start_tokens),
                end_token=1
                )

        # Decoder is partially based on @ilblackdragon//tf_example/seq2seq.py
        with tf.variable_scope(scope, reuse=reuse):
            # attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            #     num_units=self.num_units, memory=encoder_outputs,
            #     memory_sequence_length=input_lengths)
            cell = tf.contrib.rnn.LSTMCell(num_units=self.num_units)
            # attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=self.num_units / 2)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(cell, self.vocab_size, reuse=reuse)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=out_cell, helper=helper,
                initial_state=encoder_state)
            outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=self.FLAGS.output_max_length + 1)

            return outputs[0]

    def seq_loss(self, decoding, actual):
            train_output = tf.concat([tf.expand_dims(self.start_tokens, 1), actual], 1)
            weights = tf.to_float(tf.not_equal(train_output[:, :-1], 1))
            # tf.identity(decoding.rnn_output[0], name='decoder_output')
            # tf.identity(actual[0], name='actual')
            # max_seq_length = tf.shape(decoding.rnn_output)[1]
            loss = tf.contrib.seq2seq.sequence_loss(
                            decoding.rnn_output,
                            actual,
                            # average_across_timesteps=True,
                            # average_across_batch=True,
                            weights=weights)
            return loss

    def sim_loss(self, enc1, enc2, label):
        scores = tf.sigmoid(tf.reduce_sum(tf.multiply(enc1, enc2), axis=1))
        loss = - tf.reduce_mean(label * tf.log(scores + .0001) + (1 - label) * tf.log(1 - scores + .001))
        return loss

    def make_graph(self,mode, features, labels, params):
        self.embed_dim = params.embed_dim
        self.num_units = params.num_units

        # Data
        source, target, label   = features['source'], features['target'], features['label']
        self.batch_size     = tf.shape(source)[0]
        self.start_tokens   = tf.zeros([self.batch_size], dtype= tf.int64)

        with tf.variable_scope('encode'):
            source_encoder_out = self.encode(source)
            target_encoder_out = self.encode(target, reuse=True)

        # Save embeddings
        with tf.variable_scope('encode/embed', reuse=True):
            self.embeddings = tf.get_variable('embeddings')

        # Decode
        train_output_source = self.decode(source_encoder_out, 'decode', source)
        train_output_target = self.decode(target_encoder_out, 'decode', target, reuse=True)
        pred_output_source = self.decode(source_encoder_out, 'decode', mode='predict', reuse=True)
        pred_output_target = self.decode(target_encoder_out, 'decode', mode='predict', reuse=True)

        # Loss
        # tf.Print(train_output_source, [train_output_source.rnn_output[0]])
        tf.Print(source, [source])
        source_loss = self.seq_loss(train_output_source, source)
        target_loss = self.seq_loss(train_output_target, target)
        sim_loss = self.sim_loss(source_encoder_out[1],
                                 target_encoder_out[1],
                                 label)

        loss = source_loss + target_loss + sim_loss
        # loss = source_loss



        ########## Debug #################################
        # return train_output_source, loss, source, target, label
        ##################################################



        tf.summary.scalar('source_loss', source_loss)
        tf.summary.scalar('target_loss', target_loss)
        tf.summary.scalar('sim_loss', sim_loss)
        eval_metrics = {
            'source_loss': tf.contrib.metrics.streaming_mean(source_loss),
            'target_loss': tf.contrib.metrics.streaming_mean(target_loss),
            'sim_loss': tf.contrib.metrics.streaming_mean(sim_loss)
            }
        # + sim_loss
        # For logging
        # tf.identity(train_outputs.sample_id[0], name='train_pred')

        # train_output = tf.concat([tf.expand_dims(self.start_tokens, 1), source], 1)
        # weights = tf.to_float(tf.not_equal(train_output[:, :-1], 1))
        # loss = tf.contrib.seq2seq.sequence_loss(train_outputs.rnn_output, source, weights=weights)
        train_op = layers.optimize_loss(
            loss, tf.train.get_global_step(),
            optimizer=params.optimizer,
            learning_rate=params.learning_rate,
            summaries=['loss', 'learning_rate'])

        # tf.identity(pred_output_source.sample_id[0], name='predict')
        tf.identity(pred_output_source.sample_id[0], name='predict')
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_output_source.sample_id,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metrics
        )
