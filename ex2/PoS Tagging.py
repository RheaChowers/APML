import pickle
import numpy as np
import random
import math
from scipy.special import logsumexp

START_STATE = '*START*'
START_WORD = '*START*'
END_STATE = '*END*'
END_WORD = '*END*'
RARE_WORD = '*RARE_WORD*'
START_SENTENCE = '*START*'
END_SENTENCE = '*END*'
FREQUENCY_THRESHOLD = 5

def load_data(data_path='PoS_data.pickle',
                 words_path='all_words.pickle',
                 pos_path='all_PoS.pickle'):
    """
    An example function for loading and printing the Parts-of-Speech data for
    this exercise.
    Note that these do not contain the "rare" values and you will need to
    insert them yourself.

    :param data_path: the path of the PoS_data file.
    :param words_path: the path of the all_words file.
    :param pos_path: the path of the all_PoS file.
    """

    with open('PoS_data.pickle', 'rb') as f:
        data = pickle.load(f)
    with open('all_words.pickle', 'rb') as f:
        words = pickle.load(f)
    with open('all_PoS.pickle', 'rb') as f:
        pos = pickle.load(f)

    # print("The number of sentences in the data set is: " + str(len(data)))
    # print("\nThe tenth sentence in the data set, along with its PoS is:")
    # print(data[10][1])
    # print(data[10][0])

    # print("\nThe number of words in the data set is: " + str(len(words)))
    # print("The number of parts of speech in the data set is: " + str(len(pos)))

    # print("one of the words is: " + words[34467])
    # print("one of the parts of speech is: " + pos[17])

    # print(pos)
    
    return data, words, pos


class Baseline(object):
    '''
    The baseline model.
    '''

    def __init__(self, pos_tags, words, training_set):
        '''
        The init function of the baseline Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        '''

        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos: i for (i, pos) in enumerate(pos_tags)}
        self.word2i = {word: i for (i, word) in enumerate(words)}
        # create probability vectors - one for pos tags - P(y_i) and one for words: P(y_i|x)
        self.pos_probs, self.word2pos = baseline_mle(training_set, self)

    def MAP(self, sentences):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        '''
        all_pos_tags = []  # the tags for all sentences
        for sent in sentences:
            sent_tags = []  # the tags for current sentence
            # iterate over all words in sentence and assign them a postag
            for i, w in enumerate(sent):
                # If the word was in the training set, select according to argmax(P(y_j)P(y_j|x_i))
                if w in self.words:
                    # find argmax P(y_i)P(y_i|x_j)
                    word_index = self.word2i[w]
                    most_prob_pos_indx = np.argmax(np.multiply(self.word2pos[:, word_index],
                                                               self.pos_probs))
                    most_prob_tag = self.pos_tags[most_prob_pos_indx]
                    sent_tags.append(most_prob_tag)
                else:  # Else, select a tag according to tag distribution
                    probable_pos = np.random.choice(self.pos_size, 1, p=self.pos_probs)
                    most_prob_tag = self.pos_tags[probable_pos[0]]
                    sent_tags.append(most_prob_tag)
            all_pos_tags.append(sent_tags)
        return all_pos_tags


def baseline_mle(training_set, model):
    """
    a function for calculating the Maximum Likelihood estimation of the
    multinomial and emission probabilities for the baseline model.

    :param training_set: an iterable sequence of sentences, each containing
            both the words and the PoS tags of the sentence (as in the "data_example" function).
    :param model: an initial baseline model with the pos2i and word2i mappings among other things.
    :return: a mapping of the multinomial and emission probabilities. You may implement
            the probabilities in |PoS| and |PoS|x|Words| sized matrices, or
            any other data structure you prefer.
    """
    # init empty arrays
    pos_probabilities = np.zeros((model.pos_size))
    word_pos_probability = np.zeros((model.pos_size, model.words_size))
    # iterate over training set and count appearances of postags, and postags per words
    for sent in training_set:
        postags, sent_words = sent
        for i, (p, w) in enumerate(zip(postags, sent_words)):
            pos_index, word_index = model.pos2i[p], model.word2i[w]
            pos_probabilities[pos_index] += 1
            word_pos_probability[pos_index, word_index] += 1
    # normalize all vectors
    word_pos_probability = word_pos_probability / np.sum(word_pos_probability, axis=0)
    pos_probabilities = pos_probabilities / np.sum(pos_probabilities)
    return pos_probabilities, word_pos_probability



def edit_input(words, pos_tags, training_set):
    """
    adds the rare word, end_sentence and start_sentence tags to the PoS
    replaces rare words with the RARE_WORD tag
    return: lists of words, pos_tags and training tags and sentences
    """
    pos_tags.extend([END_SENTENCE, START_SENTENCE])
    word_count = {w: 0 for w in words}
    word_count[RARE_WORD] = 0
    for (pos, sentence) in training_set:
        for word in sentence:
            word_count[word] += 1
    # new_words = [w for w in words if word_count[w] > FREQUENCY_THRESHOLD]
    new_words = set([RARE_WORD])
    for (pos, sentence) in training_set:
        for i, word in enumerate(sentence):
            # word isn't rare - add it to the set of words and don't replace it
            if word_count[word] >= FREQUENCY_THRESHOLD:
                new_words.add(word)
            else:  # replace the word with a "RARE_WORD" sign
                sentence[i] = RARE_WORD
    return list(new_words), pos_tags, training_set


class HMM(object):
    '''
    The basic HMM_Model with multinomial transition functions.
    '''

    def __init__(self, pos_tags, words, training_set):
        '''
        The init function of the basic HMM Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        '''
        self.words, self.pos_tags, training_set = edit_input(words, pos_tags, training_set)
        # self.words = words
        # self.pos_tags = pos_tags
        self.word_set = set(self.words)
        self.words_size = len(self.words)
        self.pos_size = len(self.pos_tags)
        self.pos2i = {pos:i for (i, pos) in enumerate(self.pos_tags)}
        self.word2i = {word:i for (i, word) in enumerate(self.words)}
        self.transition, self.emission = hmm_mle(training_set, self)

    def sample(self, n):
        '''
        Sample n sequences of words from the HMM.
        :return: A list of word sequences.
        '''

        # P(x_1,x_2,...x_n)P(x_1|y_1)xP(y_1)xP(x_2|y_2)xP(y_2)x...xP(x_n|y_n)xP(y_n)
        # sampling will word in the following way: sample a tag according to
        # transition probabilities, then sample a word according to the emission probabilities.
        sequences = []
        for i in range(n):
            sentence = []
            cur_index = self.pos2i[START_SENTENCE]
            # while we didn't choose the END_SENTENCE tag as our next tag, continue choosing words
            while cur_index != self.pos2i[END_SENTENCE]:
                # randomly choose a tag index according to the transition function
                # cur_index = np.random.choice(self.pos_size, 1, p=np.exp(self.transition[cur_index, :]))[0]
                cur_index = np.random.choice(self.pos_size, 1, p=self.transition[cur_index, :])[0]
                if cur_index == self.pos2i[END_SENTENCE]:
                    break
                # randomly choose a word according to the emission function
                # prob_word = np.random.choice(self.words_size, 1, p=np.exp(self.emission[cur_index, :]))[0]
                prob_word = np.random.choice(self.words_size, 1, p=self.emission[cur_index, :])[0]
                chosen_word = self.words[prob_word]
                sentence.append(chosen_word)
            sequences.append(sentence)
        return sequences

    def viterbi(self, sentences):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        '''
        all_pos_tags = []
        # use log probabilities for numerical reasons, its ok we have log(0) - we'll get -inf
        self.transition = np.log(self.transition)
        self.emission = np.log(self.emission)

        for sent in sentences:  # iterate over all sentences
            sentence_length = len(sent)
            pos_tags = [""]*sentence_length
            # dynamic programming tables for probabilities and backpointers to retrace tags
            # viterbi contains +1 rows for start_sentence tags
            viterbi = np.full((sentence_length+1, self.pos_size), -np.inf)
            back_pointers = np.zeros((sentence_length, self.pos_size), dtype=int)
            # since the 0th tag is START and log(1)=0:
            viterbi[0, self.pos2i[START_SENTENCE]] = 0
            for k in range(1, sentence_length + 1):  # for every word in sentence update the word's row
                cur_word = sent[k-1]  # the first word is with index 0 but we gave that row to START
                # assign the word its index, and if we dont recognize it give it 'rare word'
                if cur_word not in self.word2i:
                        w = self.word2i[RARE_WORD]
                else:  # else treat it like a rare word
                    w = self.word2i[cur_word]
                # generate a probability vector for each PosTag and select the tag which maximizes    
                all_cur_probabilies = viterbi[k-1, :].reshape(-1, 1) + self.emission[:, w] + self.transition
                back_pointers[k-1, :] = np.argmax(all_cur_probabilies, axis=0)
                viterbi[k, :] = np.max(all_cur_probabilies, axis=0)

            last_tag = np.argmax(np.add(viterbi[sentence_length, :], self.transition[:, self.pos2i[END_SENTENCE]]))
            # iterate over backpointers table and extract the postags
            pos_tags[-1] = self.pos_tags[last_tag]
            for k in range(sentence_length-1, 0, -1):
                last_tag = back_pointers[k, last_tag]
                pos_tags[k-1] = self.pos_tags[last_tag]
            all_pos_tags.append(pos_tags)
        return all_pos_tags


def hmm_mle(training_set, model):
    """
    a function for calculating the Maximum Likelihood estimation of the
    transition and emission probabilities for the standard multinomial HMM.

    :param training_set: an iterable sequence of sentences, each containing
            both the words and the PoS tags of the sentence (as in the "data_example" function).
    :param model: an initial HMM with the pos2i and word2i mappings among other things.
    :return: a mapping of the transition and emission probabilities. You may implement
            the probabilities in |PoS|x|PoS| and |PoS|x|Words| sized matrices, or
            any other data structure you prefer.
    """
    # assume that START \ END sentence are in the postags of the model
    transition = np.zeros((model.pos_size, model.pos_size))
    emission = np.zeros((model.pos_size, model.words_size))
    start_index = model.pos2i[START_SENTENCE]
    end_index = model.pos2i[END_SENTENCE]
    for sample in training_set:
        tags, words = sample
        for i, (p, w) in enumerate(zip(tags, words)):
            tag_index = model.pos2i[p]
            word_index = model.word2i[w]
            emission[tag_index, word_index] += 1
            if i == 0:  # first tag
                transition[start_index, tag_index] += 1
            else:
                last_tag_index = model.pos2i[tags[i-1]]
                transition[last_tag_index, tag_index] += 1
            if i == len(words) - 1:  # last tag
                transition[tag_index, end_index] += 1
    # normalize and take log probabilities
    t_row_sum = np.sum(transition, axis=1)
    transition = (transition.T * np.divide(1, t_row_sum, where=t_row_sum!=0)).T
    e_row_sum = np.sum(emission, axis=1)
    emission = (emission.T * np.divide(1, e_row_sum, where=e_row_sum!=0)).T
    return transition, emission


class MEMM(object):
    '''
    The base Maximum Entropy Markov Model with log-linear transition functions.
    '''

    def __init__(self, pos_tags, words, training_set, phi=None):
        '''
        The init function of the MEMM.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        :param phi: the feature mapping function, which accepts two PoS tags
                    and a word, and returns a list of indices that have a "1" in
                    the binary feature vector.
        '''

        self.words, self.pos_tags, training_set = edit_input(words, pos_tags, training_set)
        self.words_size = len(self.words)
        self.pos_size = len(self.pos_tags)
        self.pos2i = {pos: i for (i, pos) in enumerate(self.pos_tags)}
        self.word2i = {word: i for (i, word) in enumerate(self.words)}
        # initialize w as a random vector of size P^2 + PxW to hold transition and emission
        dimension = self.pos_size*(self.pos_size + self.words_size)
        self.w = np.random.normal(loc=0, scale=0.1, size=dimension).astype(np.float32)
        # probabilities
        self.phi = self.get_phi(training_set)
        # train w using phi
        self.w = perceptron(training_set, self, self.w)


    def viterbi(self, sentences, weights):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of POS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :param w: a dictionary that maps a feature index to it's weight.
        :return: iterable sequence of POS tag sequences.
        '''
        all_pos_tags = []
        for s, sent in enumerate(sentences):  # iterate over all sentences
            sentence_length = len(sent)
            pos_tags = [""] * sentence_length
            # dynamic programming tables for probabilities and backpointers to retrace tags
            # viterbi contains +1 rows for start_sentence tags
            viterbi = np.zeros((sentence_length+1, self.pos_size))
            back_pointers = np.zeros((sentence_length, self.pos_size), dtype=int)
            # since the 0th tag is START:
            viterbi[0, self.pos2i[START_SENTENCE]] = 1
            # precalculate all log Z's required for the sentence to save time
            Z = self.calc_log_Z(weights, sent)
            for k in range(1, sentence_length + 1):  # for every word in sentence update the word's row
                cur_word = sent[k-1]  # the first word is with index 0 but we gave that row to START
                # assign the word its index, and if we dont recognize it give it 'rare word'
                if cur_word not in self.word2i:
                        word_id = self.word2i[RARE_WORD]
                        sent[k-1] = RARE_WORD
                else:
                    word_id = self.word2i[cur_word]
                # chose the best PoS tag; backpointers doesnt contain +1 rows so use k-1
                w_stack = []
                for p in range(self.pos_size):
                    w_sum = self.calc_w_at(p, word_id, weights, cur_tag=True)
                    w_stack.append(w_sum)
                all_cur_probabilies = viterbi[k-1, :].reshape(-1, 1) + w_stack - Z[k-1, :]
                back_pointers[k-1, :] = np.argmax(all_cur_probabilies, axis=0)
                viterbi[k, :] = np.max(all_cur_probabilies, axis=0)

            last_tag = np.argmax(np.add(
                viterbi[sentence_length, :], weights[self.phi(self, END_SENTENCE, -1, -1)]))
            # iterate over backpointers table and extract the postags
            pos_tags[-1] = self.pos_tags[last_tag]
            for k in range(sentence_length-1, 0, -1):
                last_tag = back_pointers[k, last_tag]
                pos_tags[k-1] = self.pos_tags[last_tag]
            all_pos_tags.append(pos_tags)
        return all_pos_tags

    def calc_log_Z(self, weights, sentence):
        sentence_length = len(sentence)
        Z = np.zeros((sentence_length, self.pos_size))
        for i, word in enumerate(sentence):
            if word in self.word2i:
                word_index = self.word2i[word]
            else:
                word_index = self.word2i[RARE_WORD]
            for p in range(self.pos_size):
                Z[i, p] = logsumexp(self.calc_w_at(p, word_index, weights, False))
        return Z

    def calc_w_at(self, tag, word, weights, cur_tag):
        """
        :param tag: an integer representing the id of a tag
        :param word: an integer representing the id of a word
        :param weights: the current weights vector
        :param cur_tag: IMPORTANT - a boolean parameter, TRUE if the given tag is y_t and false if it 
        is y_t-1 - this tells us if the iteration over all possible tags is for any NEXT tag or 
        PREVIOUS tag
        :return: a vector, where the i'th place is the inner product <w,phi> for 
        phi of (y_i,tag,word) or (tag,y_i,word), depending on the value of t
        """
        weights_sum_vector = np.zeros((self.pos_size))  # an empty vector which we fill
        for i in range(self.pos_size):
            # calculate the indices of w which we use using phi
            if cur_tag:  # if t==True then the input tag is y_{t}
                indices = self.phi(self, tag, i, word)
            else:  # else the input tag is y_{t-1}
                indices = self.phi(self, i, tag, word)
            weights_sum_vector[i] = weights[indices].sum()
        return weights_sum_vector

    def get_phi(self, training_set):
        """
        taks the training set and returns a function phi_indices which returns the relevant
        indices to a given y_t, y_{t-1} and word. These are used later to get the relevant
        weights from w.
        """
        # get the transition and emission functions from the HMM MLE. We will treat them 
        # as 0,1 vectors (and not probability vectors as they really are, since w will learn the
        # probability function).
        transition, emission = hmm_mle(training_set, self)

        def phi_indices(self, tag_t, tag_t_minus_one=-1, word=-1):
            """
            First P^2 indices are transitions which is of size PxP
            Second P*W indices are for emission which is of size PxW
            """
            indices = []
            if type(word) is str:
                if word in self.word2i:
                    word = self.word2i[word]
                else:
                    word = self.word2i[RARE_WORD]
            if type(tag_t) is str:
                tag_t = self.pos2i[tag_t]
            if type(tag_t_minus_one) is str:
                tag_t_minus_one = self.pos2i[tag_t_minus_one]
            if tag_t_minus_one == -1 and word == -1:  # we want the indices relevant to END_SENTENCE
                return [self.pos_size*i + tag_t for i in range(self.pos_size)]
            if transition[tag_t_minus_one, tag_t] > 0.0:
                indices.append(self.pos_size*tag_t_minus_one + tag_t)    
            if word != -1:  # we have a word, we are not at an end of a sentence
                if emission[tag_t, word] > 0.0:
                    indices.append(self.pos_size**2 + self.words_size * tag_t + word)
            return indices
        return phi_indices


def perceptron(training_set, initial_model, w0, eta=0.1, epochs=1):
    """
    learn the weight vector of a log-linear model according to the training set.
    :param training_set: iterable sequence of sentences and their parts-of-speech.
    :param initial_model: an initial MEMM object, containing among other things
            the phi feature mapping function.
    :param w0: an initial weights vector.
    :param eta: the learning rate for the perceptron algorithm.
    :param epochs: the amount of times to go over the entire training data (default is 1).
    :return: w, the learned weights vector for the MEMM.
    """
    w = w0
    num_samples = len(training_set)
    for epoch in range(epochs):
        random.shuffle(training_set)
        for i, sample in enumerate(training_set):
            pos_tags, sentence = sample
            # every 1000 samples check we are learning by sampling 32 random samples, running viterbi
            # on them and checking the accuracy
            if i % 1000 == 0:
                print("Perceptron - Sample Number ", i)
                random_sample = random.sample(training_set, 64)
                total, correct = 0, 0
                for smp in random_sample:
                    t, s = smp
                    pred = initial_model.viterbi([s], w)[0]
                    for j, q in enumerate(pred):
                        total += 1
                        if q == t[j]:
                            correct += 1
                print("the current accuracy is: ", correct/total)
            y_hat = initial_model.viterbi([sentence], w)[0]  # get the current model's predictions
            # calculate the difference between phi of the original postags and predicted
            # set delta_y for the first tag (since the 0 tag is START_SENTENCE)
            w[initial_model.phi(initial_model, pos_tags[0], START_SENTENCE, sentence[0])] += eta
            w[initial_model.phi(initial_model, y_hat[0], START_SENTENCE, sentence[0])] -= eta
            for t in range(1, len(sentence)):
                w[initial_model.phi(initial_model, pos_tags[t], pos_tags[t-1], sentence[t])] += eta
                w[initial_model.phi(initial_model, y_hat[t], y_hat[t-1], sentence[t])] -= eta
            w[initial_model.phi(initial_model, END_SENTENCE, pos_tags[t-1], -1)] += eta
            w[initial_model.phi(initial_model, END_SENTENCE, y_hat[t-1], -1)] -= eta
        w /= num_samples  # divide by number of samples to get the average
    return w


def train_and_test(model_name):
    """
    Trains the model according to the given model name
    and prints the accuracy
    """
    data, words, pos = load_data()
    random.shuffle(data)
    samps = math.ceil(len(data) * 0.9)
    train = data[:samps]
    test = data[samps:]
    test_sentences = [s[1] for s in test]
    test_tags = [s[0] for s in test]
    if model_name == "baseline":
        b = Baseline(pos, words, train)
        total, correct = 0, 0
        for i, tag in enumerate(b.MAP(test_sentences)):
            for j, t in enumerate(tag):
                total += 1
                if tag[j] == test_tags[i][j]:
                    correct += 1
        print("the baseline accuracy is :", correct/total)
    if model_name == "hmm":
        mod = HMM(pos, words, train)
        for k in mod.sample(5):
            print(k)
        total, correct = 0, 0
        for i, tag in enumerate(mod.viterbi(test_sentences)):
            for j, t in enumerate(tag):
                total += 1
                if tag[j] == test_tags[i][j]:
                    correct += 1
        print("the hmm accuracy is :", correct/total)
    if model_name == "memm":
        for p in [0.9]:
            train = data[:math.ceil(len(data) * p)]
            m = MEMM(pos, words, train)
            total, correct = 0, 0
            for i, tag in enumerate(m.viterbi(test_sentences, m.w)):
                for j, t in enumerate(tag):
                    total += 1
                    if tag[j] == test_tags[i][j]:
                        correct += 1
            print("the MEMM accuracy is :", correct/total, " when using ", 100*p, "% of the data")


if __name__ == '__main__':
    # train_and_test("baseline")
    train_and_test("hmm")
    # train_and_test("memm")
