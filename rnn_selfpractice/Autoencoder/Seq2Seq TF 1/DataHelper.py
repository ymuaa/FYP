import numpy as np

def random_sequences(length_min, length_max, vocab_min, vocab_max, batch_size):
    '''
    generate random input batch

    :param length_min:  sentence length (lower bound)
    :param length_max:  sentence length (higher bound)
    :param vocab_min:   vocab range (lower bound)
    :param vocab_max:   vocab range (higher bound)
    :param batch_size:  #sentences per batch
    :return: such a sentence batch
    '''
    def random_length():
        if length_min == length_max:
            return length_min
        return np.random.randint(length_min, length_max + 1)

    while True:
        yield[
            np.random.randint(low=vocab_min, high=vocab_max, size=random_length()).tolist()
            for _ in range(batch_size)
        ]

def make_batch(inputs, max_sequence_length=None):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths