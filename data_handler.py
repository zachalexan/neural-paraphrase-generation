import tensorflow as tf
import numpy as np
import re
from collections import defaultdict



class Data:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        # create vocab and reverse vocab maps
        self.vocab     = {}
        self.rev_vocab = {}
        self.END_TOKEN = 1
        self.UNK_TOKEN = 2
        with open(FLAGS.vocab_filename) as f:
            for idx, line in enumerate(f):
                self.vocab[line.strip()] = idx
                self.rev_vocab[idx] = line.strip()
        self.vocab_size = len(self.vocab)
        self.remove_word_prob = FLAGS.remove_word_prob
        self.swap_words_prob = FLAGS.swap_words_prob

    def _random_word_vec(self, dim):
        x = [np.random.laplace() for i in range(dim)]
        x = np.clip(np.array(x) * .4, -1, 1)
        return x

    def initialize_word_vectors(self):
        # Load GLOVE vectors
        words = set(self.vocab.keys())
        embeddings_dict = {}
        with open(self.FLAGS.word_vectors) as f:
            for line in f:
                vec = line.split()
                word = vec[0]
                if word in words:
                    coefs = np.asarray(vec[1:], dtype='float64')
                    embeddings_dict[word] = coefs
        embed_words = set(embeddings_dict.keys())

        # Initialize matrix
        self.embeddings_mat = np.zeros((self.vocab_size, self.FLAGS.embed_dim))
        for key, val in self.vocab.iteritems():
            if key in embed_words:
                self.embeddings_mat[val] = embeddings_dict[key]
            else:
                self.embeddings_mat[val] = self._random_word_vec(self.FLAGS.embed_dim)
        self.embeddings_mat = np.transpose(self.embeddings_mat)

    def tokenize_and_map(self, line, mode='train', remove_word_prob = 0, swap_words_prob = 0):
        tokens = [self.vocab.get(token, self.UNK_TOKEN) for token in line.split()]

        if mode == 'train':
            # Remove some words at random
            if remove_word_prob > 0:
                n = len(tokens)
                if n > 3:
                    num_to_rmv = np.random.binomial(n, remove_word_prob)
                    idx_to_rmv = np.random.choice(n, num_to_rmv, replace=False)
                    tokens = [t for i, t in enumerate(tokens) if not i in idx_to_rmv]

            # Swap the order of two words at random
            if swap_words_prob > 0:
                n = len(tokens) - 1
                if n > 3:
                    num_to_swap = np.random.binomial(n, swap_words_prob)
                    idx_to_swap = np.random.choice(n, num_to_swap, replace=False)
                    for i in idx_to_swap:
                        tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]

        return tokens


    def make_input_fn(self, mode='train'):
        def input_fn():
            source_in = tf.placeholder(tf.int64, shape=[None, None], name='source_in')
            source_out = tf.placeholder(tf.int64, shape=[None, None], name='source_out')
            target_in = tf.placeholder(tf.int64, shape=[None, None], name='target_in')
            target_out = tf.placeholder(tf.int64, shape=[None, None], name='target_out')
            label = tf.placeholder(tf.float32, shape=[None,], name='label')
            tf.identity(source_in[0], 'source')
            # tf.identity(output[0], 'target_ex')
            return {'source_in': source_in,
                    'source_out': source_out,
                    'target_in': target_in,
                    'target_out': target_out,
                    'label': label}, None

        def sampler(mode='train'):
            epoch = 1
            file1, file2, file3 = self.FLAGS.input_filename, self.FLAGS.output_filename, self.FLAGS.shuffled_filename
            if mode == 'test':
                file1, file2, file3 = (re.sub('train', 'test', f) for f in (file1, file2, file3))
            while True:
                print 'Start of Epoch ' + str(epoch)
                epoch += 1
                with open(file1) as finput, \
                     open(file2) as foutput, \
                     open(file3) as fshuffled:
                         for source, target, shuffled in zip(finput, foutput, fshuffled):
                             label = 1
                             if np.random.rand() < .5:
                                 target = shuffled
                                 label = 0
                             if max(len(source.split()), len(target.split())) > self.FLAGS.max_length:
                                 continue
                             source_in = self.tokenize_and_map(source,
                                                               mode=mode,
                                                               remove_word_prob=self.remove_word_prob,
                                                               swap_words_prob=self.swap_words_prob) + [self.END_TOKEN]
                             target_in = self.tokenize_and_map(target,
                                                               mode=mode,
                                                               remove_word_prob=self.remove_word_prob,
                                                               swap_words_prob=self.swap_words_prob) + [self.END_TOKEN]
                             if label & (np.random.rand() < .5):
                                 target_out = self.tokenize_and_map(source, mode='test') + [self.END_TOKEN]
                                 source_out = self.tokenize_and_map(target, mode='test') + [self.END_TOKEN]
                             else:
                                 source_out = self.tokenize_and_map(source, mode='test') + [self.END_TOKEN]
                                 target_out = self.tokenize_and_map(target, mode='test') + [self.END_TOKEN]
                             yield {
                                'source_in': source_in,
                                'source_out': source_out,
                                'target_in': target_in,
                                'target_out': target_out,
                                'label': label
                                }

        data_feed = sampler(mode=mode)

        def feed_fn():
            # source, target, label = [], [], []
            # input_length, output_length = 0, 0
            # max_length = 0

            label = []
            phrases = defaultdict(list)
            keys = ['source_in', 'source_out', 'target_in', 'target_out']
            lengths = defaultdict(int)
            for i in range(self.FLAGS.batch_size):
                rec = data_feed.next()
                label.append(rec['label'])
                for key in keys:
                    phrases[key].append(rec[key])
                    lengths[key] = max(lengths[key], len(phrases[key][-1]))
                # source.append(rec['source'])
                # target.append(rec['target'])
                # input_length = max(input_length, len(source[-1]))
                # output_length = max(output_length, len(target[-1]))
                # max_length = max(max_length, len(source[-1]), len(target[-1]))
            for i in range(self.FLAGS.batch_size):
                for key in keys:
                    phrases[key][i] += [self.END_TOKEN] * (lengths[key] - len(phrases[key][i]))
                # source[i] += [self.END_TOKEN] * (input_length - len(source[i]))
                # target[i] += [self.END_TOKEN] * (output_length - len(target[i]))
            return {
                'source_in:0': phrases['source_in'],
                'source_out:0': phrases['source_out'],
                'target_in:0': phrases['target_in'],
                'target_out:0': phrases['target_out'],
                'label:0': label
                }
        return input_fn, feed_fn

    def get_formatter(self,keys):
        def to_str(sequence):
            try:
                tokens = [self.rev_vocab.get(x, "<UNK>") for x in sequence.tolist()]
            except:
                tokens = [self.rev_vocab.get(x, "<UNK>") for x in sequence[0].tolist()]
            return ' '.join(tokens)

        def format(values):
            res = []
            for key in keys:
                res.append("****%s == %s" % (key, to_str(values[key]).replace('</S>','').replace('<S>', '')))
            return '\n'+'\n'.join(res)
        return format
