import math


def _get_lyrics(fname):
    lyrics = []
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            lyrics.extend(line)
    lyrics = lyrics[:-9]  # getting rid of warning at end of each lyrics sample
    lyrics = [''.join(e.lower() for e in x if e.isalpha()) for x in
              lyrics]  # remove everything that is not a letter and convert all letters to lowercase
    lyrics = [x for x in lyrics if x not in ['']]  # remove all invalid words
    return lyrics


def _get_word_counts(X):
    word_counts = {}
    for lyrics_pair in X:
        for lyrics_string in lyrics_pair:
            for word in lyrics_string:
                if word in word_counts.keys():
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
    return word_counts


def _count_threshold_filter(X, threshold, word_counts):
    for i in range(len(X)):
        X[i][0] = [x for x in X[i][0] if word_counts[x] >= threshold]
        X[i][1] = [x for x in X[i][1] if word_counts[x] >= threshold]
    for key in word_counts.keys():
        if not word_counts[key] >= threshold:
            del word_counts[key]
    return X, word_counts


def get_data(pair_fname, lyrics_path, num_examples=float('Inf'), threshold=-1, n_class=5):
    X = []
    y = []
    with open(pair_fname, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            # y.append(float(line[2]))
            y.append(math.ceil(float(line[2]) * n_class))
            if y[-1] == n_class + 1:
                y[-1] = n_class
            x = [[], []]
            x[0] = _get_lyrics(lyrics_path + line[0] + '.txt')
            x[1] = _get_lyrics(lyrics_path + line[1] + '.txt')
            X.append(x)
    word_counts = _get_word_counts(X)
    if threshold > 0:
        X, word_counts = _count_threshold_filter(X, threshold, word_counts)
    max_steps = max([max(len(x[0]), len(x[1])) for x in X])
    return X, y, word_counts, max_steps
