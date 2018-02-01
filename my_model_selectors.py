import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant
    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        minBIC = np.inf
        best_model = None

        for i in range(self.min_n_components, self.max_n_components + 1):
            try:
                test_model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                                                random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = test_model.score(self.X, self.lengths)
                logN = math.log(len(self.X))
                # Really no idea of how to calculate parameters, got help from forums:
                # https://discussions.udacity.com/t/verifing-bic-calculation/246165
                # but it doesn't mean I understand it... it just works.
                p = i * i + 2 * i * len(self.X[0]) - 1
                # Calculate BIC = -2 * logL + p * logN
                # where L is the likelihood of the fitted model, p is the number of parameters, and N
                # is the number of data points. 
                test_bic = -2 * logL + p * logN
                if test_bic < minBIC:
                    minBIC = test_bic
                    best_model = test_model
            except:
                pass
        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Reimplemented following the comments from the reviewer
        # This is the pseudocode:
        # best_dic <- float("-Inf")    # Initialize best_model and best_dic
        # best_model <- None
        # for each value of n_components:
        #     model <- self.base_model(n)
        #     logL <- model.score(self.X, self.lengths)
        #     penalty <- np.mean( [ model.score(self.hwords[word]) for word in self.words if word != self.this_word ] )
        #     dic <- logL - penalty
        #     if dic > best_dic:
        #         best_model <- model
        #         best_dic <- dic
        # return best_model

        maxDIC = -np.inf
        best_model = None

        for i in range(self.min_n_components, self.max_n_components + 1):
            try:
                penalty = []
                model = self.base_model(i)
                logL = model.score(self.X,self.lengths)
                for word in self.words:
                    if word != self.this_word:
                            penalty.append(model.score(self.hwords[word]))
                penaltyAvg = np.mean(penalty)
                dic = logL - penaltyAvg

                if dic > maxDIC:
                    maxDIC = dic
                    best_model = model
            except:
                pass
            
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # FISH was returning wrong values, number of splits seems cannot be more than 2, so got help here:
        # https://discussions.udacity.com/t/issue-with-selectorcv/299868
        # but this solution doesn't work on part 3.
        # n_splits = min(len(self.lengths), 3)
        n_splits = 2
        split_method = KFold(n_splits)
        final_score = -np.inf
        best_model = None

        # For each number of n_components...
        for i in range(self.min_n_components, self.max_n_components + 1):
            try:
                # We do a cross validation
                sum_score = 0
                denominator = 0
                average = 0

                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    # Train and test dataset split
                    self.train_X, self.train_lengths = combine_sequences(cv_train_idx,self.sequences)
                    self.test_X, self.test_lengths = combine_sequences(cv_test_idx,self.sequences)
                    test_model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(self.train_X, self.train_lengths)
                    # Get the score and calculate the average
                    denominator += 1
                    sum_score = sum_score + test_model.score(self.test_X, self.test_lengths)
                average = sum_score / denominator
                # If average is better, this model is also better than the previous ones
                if average > final_score:
                    final_score = average
                    # Return a model trained with all the data
                    best_model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            except:
                pass

        return best_model