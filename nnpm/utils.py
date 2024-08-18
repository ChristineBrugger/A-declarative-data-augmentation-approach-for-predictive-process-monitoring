"""
This script is based on the following source code:
    https://github.com/nicoladimauro/nnpm
    https://gitlab.citius.usc.es/efren.rama/pmdlcompararator
We just adjusted some parts to efficiently use it in our study.
"""

def load_data(logfile=None, max_len=None, parsed_vocabulary=None, y_dict=None, min_prefix_length=2):

    import datetime
    import time
    import numpy as np
    import csv
    from datetime import datetime
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    vocabulary = set()

    csvfile = open(logfile, 'r')
    logreader = csv.reader(csvfile, delimiter=',')
    next(logreader, None)  # skip the headers

    lastcase = '' 
    casestarttime = None
    lasteventtime = None
    firstLine = True

    lines = [] # these are all the activity seq
    timeseqs = [] # time sequences (differences between two events)

    numcases = 0
    max_length = 0

    for row in logreader:
        t = datetime.strptime(row[2], "%Y/%m/%d %H:%M:%S.%f")
        if row[0] != lastcase:  # 'lastcase' is to save the last executed case for the loop
            casestarttime = t
            lasteventtime = t
            lastcase = row[0]
            if not firstLine:
                lines.append(line)
                timeseqs.append(times)
                if len(line) > max_length:
                    max_length = len(line)
            line = []
            times = []
            numcases += 1

        vocabulary.add(row[1])
        line.append(row[1])
        timesincelastevent = t - lasteventtime
        timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds + timesincelastevent.microseconds / 1000000
        # +1 avoid zero
        times.append(timediff + 1)
        lasteventtime = t
        firstLine = False

    lines.append(line)
    timeseqs.append(times)

    if parsed_vocabulary is None:
        vocabulary = {key: idx for idx, key in enumerate(vocabulary)}
    else:
        vocabulary = parsed_vocabulary

    divisor = np.mean([item for sublist in timeseqs for item in sublist]) # average time between events
    numcases += 1
    print("----")
    print("Num cases: ", numcases)

    if len(line) > max_length:
        max_length = len(line)

    X = []
    X1 = []
    y = []
    y_t = []

    max_length = 0
    prefix_sizes = []
    seqs = 0
    vocab = set()
    for seq, time in zip(lines, timeseqs):
        code = []
        code.append(vocabulary[seq[0]])
        code1 = []
        code1.append(np.log(time[0] + 1))

        code.append(vocabulary[seq[1]])
        code1.append(np.log(time[1] + 1))

        vocab.add(seq[0])
        vocab.add(seq[1])

        for i in range(2, len(seq)):
            prefix_sizes.append(len(code))

            if len(code) > max_length:
                max_length = len(code)
            X.append(code[:])
            X1.append(code1[:])
            y.append(vocabulary[seq[i]])
            y_t.append(time[i] / divisor)

            code.append(vocabulary[seq[i]])
            code1.append(np.log(time[i] + 1))
            seqs += 1

            vocab.add(seq[i])

    prefix_sizes = np.array(prefix_sizes)

    print("Num sequences:", seqs)

    print("Activities: ", vocab)
    vocab_size = len(vocab)

    X = np.array(X)
    X1 = np.array(X1)
    y = np.array(y)
    y_t = np.array(y_t)

    y_unique = np.unique(y)
    if y_dict is None:
        dict_y = {}
        i = 0
        for el in y_unique:
            dict_y[el] = i
            i += 1
    else:
        dict_y = y_dict
    for i in range(len(y)):
        y[i] = dict_y[y[i]]
    y_unique = np.unique(y, return_counts=True)
    print("Classes: ", y_unique)
    n_classes = y_unique[0].shape[0]
    # padding

    # If a maximum length is provided, ignore the calculations
    if max_len is not None:
        max_length = max_len
    padded_X = pad_sequences(X, maxlen=max_length, padding='pre', dtype='float64')
    padded_X1 = pad_sequences(X1, maxlen=max_length, padding='pre', dtype='float64')
    print("----")

    return ( (padded_X, padded_X1), (y, y_t), vocab_size, max_length, n_classes, divisor, prefix_sizes, vocabulary, y_dict)
