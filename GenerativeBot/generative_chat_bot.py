import tensorflow as tf
import numpy as np
import os
import pickle

import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from string import punctuation

import tensorflow.keras as tk

import requests


class GenChatBot:
    def _init_text_content(self):
        text = open(self.content_path, encoding='utf-8').read()
        text = text.lower()
        text = text.translate(str.maketrans('', '', punctuation))

        return text

    def _get_content_stats(self):
        return {
            "chars": sorted(self.text),
            "#_chars": len(self.text),
            "unique_chars": ''.join(sorted(set(self.text))),
            "#_unique_chars": len(set(self.text))
        }

    def _to_int_text(self, dictionary, text):
        encoded_text = np.array([dictionary[c] for c in text])
        return encoded_text

    def split_sample(self, sample):
        # example :
        # sequence_length is 10
        # sample is "python is a great pro" (21 length)
        # ds will equal to ('python is ', 'a') encoded as integers
        ds = tf.data.Dataset.from_tensors((sample[:self.seq_len], sample[self.seq_len]))
        for i in range(1, (len(sample) - 1) // 2):
            # first (input_, target) will be ('ython is a', ' ')
            # second (input_, target) will be ('thon is a ', 'g')
            # third (input_, target) will be ('hon is a g', 'r')
            # and so on
            input_ = sample[i: i + self.seq_len]
            target = sample[i + self.seq_len]
            # extend the dataset with these samples by concatenate() method
            other_ds = tf.data.Dataset.from_tensors((input_, target))
            ds = ds.concatenate(other_ds)
        return ds

    def one_hot_sample(self, input_, target):
        # Returns vectors which look like [ 0, 0, 0, 1, 0 ]
        # This example is if character d is encoded as 3 and there are 5 values in total (index 0)

        unique_chars = self.text_data['#_unique_chars']
        print(type(unique_chars), type(target))
        return tf.one_hot(input_, unique_chars), tf.one_hot(target, unique_chars)

    def __init__(self, basename):
        os.chdir("../GenerativeBot/GenerativeData")

        self.BASENAME = basename

        # Change input style later.
        self.content = requests.get(
            'http://www.gutenberg.org/cache/epub/11/pg11.txt'
        ).text
        # Gets the data to understand

        # Gets data info for training data
        self.seq_len = 100
        self.batch_size = 128
        self.epoch = 30

        __check_result_path = f'results/{self.BASENAME}-{self.seq_len}.h5'

        # Sets content path to access data
        self.content_path = "data.txt"

        # Makes uniform text formatting
        self.text = self._init_text_content()

        # Gets some information from the text and makes it into an easily accessible dataform
        self.text_data = self._get_content_stats()

        # If file is already generated
        if os.path.isfile(__check_result_path):
            print("File found!")

            self.char_int_dict = pickle.load(open(f"/dicts/{self.BASENAME}-char2int.pickle", "rb"))
            self.int_char_dict = pickle.load(open(f"/dicts/{self.BASENAME}-char2int.pickle", "rb"))

            self.model = Sequential([
                LSTM(256, input_shape=(self.seq_len, self.text_data['#_unique_chars']), return_sequences=True),
                Dropout(0.3),
                LSTM(256),
                Dense(self.text_data['#_unique_chars'], activation="softmax"),
            ])

            self.model = self.model.load_weights(f"results/{self.BASENAME}-{self.seq_len}.h5")

        else:
            input("Didn't find file!")

            input("SOTP HERE!")

            # Creates dictionaries for char to int / int to char from the text_data
            self.char_int_dict = {c: i for i, c in enumerate(self.text_data['unique_chars'])}
            self.int_char_dict = {i: c for i, c in enumerate(self.text_data['unique_chars'])}

            # Dump the dictionaries into pickle files for storage.
            # Set the directory to the correct file so that items can be stored correctly.

            pickle.dump(self.char_int_dict, open(f"dicts/{self.BASENAME}-char2int.pickle", "wb"))
            pickle.dump(self.int_char_dict, open(f"dicts/{self.BASENAME}-int2char.pickle", "wb"))

            # "Encodes" the text using a function defined earlier in order to turn it into numbers (machine-readable)
            encoded_text = np.array([self.char_int_dict[c] for c in self.text])

            print(f"encoded {encoded_text}")

            # Creates a new char dataset which contains every single character from the provided text
            self.char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)

            print(f"dataset {self.char_dataset}")

            # TODO Testing code which can be deleted at a later date.
            # [print(i.numpy(), self.char_dataset[i.numpy]) for i in self.char_dataset[:20]]
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - #

            # Sequence built by batching THIS IS FLOAT WTF!
            sequences = self.char_dataset.batch(2 * self.seq_len + 1, drop_remainder=True)

            #        for sequence in sequences.take(5):
            #           print(''.join([self.int_char_dict[i] for i in sequence.numpy()]))

            print(f"SEQUENCES {sequences}")

            self.dataset = sequences.flat_map(self.split_sample)

            print(self.one_hot_sample)
            print(self.dataset)

            # Error Here :(
            self.dataset = self.dataset.map(self.one_hot_sample)

            # Repeats and shuffles the dataset drop_remainder is true since remaining samples with less size than the
            # batch_size should be eliminated.

            self.ds = self.dataset.repeat().shuffle(1024).batch(self.batch_size, drop_remainder=True)

            self.model = Sequential([
                LSTM(256, input_shape=(self.seq_len, self.text_data['#_unique_chars']), return_sequences=True),
                Dropout(0.3),
                LSTM(256),
                Dense(self.text_data['#_unique_chars'], activation="softmax"),
            ])

            model_weights_path = f"results/{self.BASENAME}-{self.seq_len}.h5"
            self.model.summary()
            self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

            if not os.path.isdir("results"):
                os.mkdir("results")
            # train the model
            self.model.fit(self.ds, steps_per_epoch=(len(encoded_text) - self.seq_len) // self.batch_size,
                           epochs=self.epoch)
            # save the model
            self.model.save(model_weights_path)

    def converse(self):
        seed = input('> ')
        chars = 1000

        gen = ''

        for i in tqdm.tqdm(range(chars), "Generating..."):
            X = np.zeros((1, self.seq_len, self.text_data['#_unique_chars']))

            for j, char in enumerate(seed):
                X[0, (self.seq_len - len(seed)) + j, self.char_int_dict[char]] = 1

            predicted = self.model.predict(X, verbose=0)[0]

            next_index = np.argmax(predicted)

            next_char = self.int_char_dict[next_index]

            gen += next_char

            seed = seed[1:] + next_char

        print(f'{self.BASENAME}: ' + gen)
