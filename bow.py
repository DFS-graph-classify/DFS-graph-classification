from collections import Counter
import json
import numpy as np


class Vocabulary:

    def __init__(self, vocabulary, wordFrequencyFilePath):
        self.vocabulary = vocabulary
        self.WORD_FREQUENCY_FILE_FULL_PATH = wordFrequencyFilePath
        self.input_word_index = {}
        self.reverse_input_word_index = {}

        self.MaxSentenceLength = None

    def PrepareVocabulary(self, reviews):
        self._prepare_Word_Frequency_Count_File(reviews)
        self._create_Vocab_Indexes()

        self.MaxSentenceLength = max([len(txt.split(" ")) for txt in reviews])

    def Get_Top_Words(self, number_words=None):
        if number_words == None:
            number_words = self.vocabulary

        chars = json.loads(open(self.WORD_FREQUENCY_FILE_FULL_PATH).read())
        counter = Counter(chars)
        most_popular_words = {key for key,
                              _value in counter.most_common(number_words)}
        return most_popular_words

    def _prepare_Word_Frequency_Count_File(self, reviews):
        counter = Counter()
        for s in reviews:
            counter.update(s.split(" "))

        with open(self.WORD_FREQUENCY_FILE_FULL_PATH, 'w') as output_file:
            output_file.write(json.dumps(counter))

    def _create_Vocab_Indexes(self):
        INPUT_WORDS = self.Get_Top_Words(self.vocabulary)

        for i, word in enumerate(INPUT_WORDS):
            self.input_word_index[word] = i

        for word, i in self.input_word_index.items():
            self.reverse_input_word_index[i] = word

    def TransformSentencesToId(self, sentences):
        vectors = []
        for r in sentences:
            words = r.split(" ")
            vector = np.zeros(len(words))

            for t, word in enumerate(words):
                if word in self.input_word_index:
                    vector[t] = self.input_word_index[word]
                else:
                    pass

            vectors.append(vector)

        return vectors

    def ReverseTransformSentencesToId(self, sentences):
        vectors = []
        for r in sentences:
            words = r.split(" ")
            vector = np.zeros(len(words))

            for t, word in enumerate(words):
                if word in self.input_word_index:
                    vector[t] = self.input_word_index[word]
                else:
                    pass
                    # vector[t] = 2 #unk
            vectors.append(vector)

        return vectors
