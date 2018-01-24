import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import tensorflow as tf
from seq2seq import Seq2seq
from data_handler import Data

from tensorflow.python import debug as tf_debug
from tensorflow.contrib.learn import learn_runner



FLAGS = tf.flags.FLAGS

# Model related
tf.flags.DEFINE_integer('num_units'         , 256           , 'Number of units in a LSTM cell')
tf.flags.DEFINE_integer('embed_dim'         , 100           , 'Size of the embedding vector')

# Training related
tf.flags.DEFINE_float('learning_rate'       , 0.001         , 'learning rate for the optimizer')
tf.flags.DEFINE_string('optimizer'          , 'Adam'        , 'Name of the train source file')
tf.flags.DEFINE_integer('batch_size'        , 32            , 'random seed for training sampling')
tf.flags.DEFINE_integer('print_every'       , 100          , 'print records every n iteration')
tf.flags.DEFINE_integer('iterations'        , 2000         , 'number of iterations to train')
tf.flags.DEFINE_string('model_dir'          , 'checkpoints' , 'Directory where to save the model')
tf.flags.DEFINE_string('experiment_dir'          , 'experiments_glove' , 'Directory where to save the experiment')

tf.flags.DEFINE_integer('input_max_length'  , 30            , 'Max length of input sequence to use')
tf.flags.DEFINE_integer('output_max_length' , 30            , 'Max length of output sequence to use')
tf.flags.DEFINE_integer('max_length' , 30            , 'Max length of output sequence to use')
tf.flags.DEFINE_bool('use_residual_lstm'    , True          , 'To use the residual connection with the residual LSTM')

# Data related
tf.flags.DEFINE_string('input_filename', 'data/mscoco/train_source.txt', 'Name of the train source file')
tf.flags.DEFINE_string('output_filename', 'data/mscoco/train_target.txt', 'Name of the train target file')
tf.flags.DEFINE_string('vocab_filename', 'data/mscoco/train_vocab.txt', 'Name of the vocab file')
tf.flags.DEFINE_string('shuffled_filename', 'data/mscoco/train_target_shuffled.txt', 'Name of shuffled targets')
tf.flags.DEFINE_string('word_vectors', '../data/glove/glove.6B.100d.txt', 'Name of word vectors file')

def run_experiment(argv=None):

    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(model_dir=FLAGS.experiment_dir)

    learn_runner.run(experiment_fn=experiment_fn,
                     run_config=run_config,
                     schedule='train_and_evaluate'
                    )

def experiment_fn(run_config, params):
    data = Data(FLAGS)
    data.initialize_word_vectors()

    model = Seq2seq(data.vocab_size, FLAGS, data.embeddings_mat)
    estimator = tf.estimator.Estimator(model_fn=model.make_graph,
#                                        model_dir=FLAGS.model_dir,
                                       config=run_config,
                                       params=FLAGS)

    train_input_fn, train_feed_fn = data.make_input_fn('train')
    eval_input_fn, eval_feed_fn = data.make_input_fn('test')

    print_vars = [
        'source',
        'predict'
        # 'decoder_output',
        # 'actual'
    ]
    print_inputs = tf.train.LoggingTensorHook(print_vars ,
                                              every_n_iter=FLAGS.print_every,
                                              formatter=data.get_formatter(['source', 'predict']))

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=FLAGS.iterations,
        min_eval_frequency=FLAGS.print_every,
        train_monitors=[tf.train.FeedFnHook(train_feed_fn), print_inputs],
        eval_hooks=[tf.train.FeedFnHook(eval_feed_fn)],
        eval_steps=10
    )
    return experiment

if __name__ == "__main__":
    tf.app.run(
        main=run_experiment
    )

# def main(args):
#     tf.logging._logger.setLevel(logging.INFO)
#
#     data  = Data(FLAGS)
#     model = Seq2seq(data.vocab_size, FLAGS)
#
#     input_fn, feed_fn = data.make_input_fn()
#     print_vars = [
#         'source_ex',
#         'target_ex',
#         'predict'
#         # 'decoder_output',
#         # 'actual'
#         ]
#     print_inputs = tf.train.LoggingTensorHook(print_vars ,
#                                               every_n_iter=FLAGS.print_every,
#                                               formatter=data.get_formatter(['source_ex', 'target_ex', 'predict']))
#
#     hooks = [
#         tf.train.FeedFnHook(feed_fn),
#         # tf_debug.LocalCLIDebugHook()
#         print_inputs
#         ]
#
#     estimator = tf.estimator.Estimator(model_fn=model.make_graph, model_dir=FLAGS.model_dir, params=FLAGS)
#     estimator.train(input_fn=input_fn, hooks=hooks, steps=FLAGS.iterations)
#
# if __name__ == "__main__":
#     tf.app.run()
