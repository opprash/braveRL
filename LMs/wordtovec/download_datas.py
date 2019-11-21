import collections
import urllib
import urllib.request
import os
import zipfile
import tensorflow as tf

def maybe_download(filename, url,expected_bytes):

    """Download a file if not present, and make sure it's the right size."""


    if not os.path.exists(filename):

        filename, _ = urllib.request.urlretrieve(url + filename, filename)

    statinfo = os.stat(filename)
    print(statinfo.st_size)

    if statinfo.st_size == expected_bytes:

       print('Found and verified', filename)

    else:


       print(statinfo.st_size)

       raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
    print(filename)
    return filename


def read_data(filename):
    print(filename)
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        return data

def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

if __name__ == '__main__':
    #url = 'http://mattmahoney.net/dc/'

    #filename = maybe_download('text8.zip', url, 31344016)
    #filename = maybe_download('text8.zip', url,284136)
    words = read_data("./text8.zip")
    print("Data size %d" % len(words))

    data, count, dictionary, reverse_dictionary = build_dataset(words,5000)

    print("Most common words (+UNK)", count[:5])

    print("Sample data", data[:10])
    del words
