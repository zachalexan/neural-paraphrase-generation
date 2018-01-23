import tensorflow as tf
import numpy as np
import re

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

    def tokenize_and_map(self,line):
        return [self.vocab.get(token, self.UNK_TOKEN) for token in line.split()]


    def make_input_fn(self, mode='train'):
        def input_fn():
            inp = tf.placeholder(tf.int64, shape=[None, None], name='source')
            output = tf.placeholder(tf.int64, shape=[None, None], name='target')
            label = tf.placeholder(tf.float32, shape=[None,], name='label')
            tf.identity(inp[0], 'source')
            # tf.identity(output[0], 'target_ex')
            return { 'source': inp, 'target': output, 'label': label}, None

        def sampler(mode='train'):
            file1, file2, file3 = self.FLAGS.input_filename, self.FLAGS.output_filename, self.FLAGS.shuffled_filename
            if mode == 'test':
                file1, file2, file3 = (re.sub('train', 'test', f) for f in (file1, file2, file3))
            while True:
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
                             yield {
                                'source': self.tokenize_and_map(source) + [self.END_TOKEN],
                                'target': self.tokenize_and_map(target) + [self.END_TOKEN],
                                'label': label
                                }

        data_feed = sampler(mode=mode)

        def feed_fn():
            source, target, label = [], [], []
            input_length, output_length = 0, 0
            # max_length = 0
            for i in range(self.FLAGS.batch_size):
                rec = data_feed.next()
                label.append(rec['label'])
                source.append(rec['source'])
                target.append(rec['target'])
                input_length = max(input_length, len(source[-1]))
                output_length = max(output_length, len(target[-1]))
                # max_length = max(max_length, len(source[-1]), len(target[-1]))
            for i in range(self.FLAGS.batch_size):
                source[i] += [self.END_TOKEN] * (input_length - len(source[i]))
                target[i] += [self.END_TOKEN] * (output_length - len(target[i]))
            return { 'source:0': source, 'target:0': target, 'label:0': label }
        return input_fn, feed_fn

    def get_formatter(self,keys):
        def to_str(sequence):
            tokens = [
                self.rev_vocab.get(x, "<UNK>") for x in sequence]
            return ' '.join(tokens)

        def format(values):
            res = []
            for key in keys:
                res.append("****%s == %s" % (key, to_str(values[key]).replace('</S>','').replace('<S>', '')))
            return '\n'+'\n'.join(res)
        return format
