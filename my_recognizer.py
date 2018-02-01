import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses

    # Get ideas and help from:
    # https://discussions.udacity.com/t/recognizer-implementation/234793/27

    for word_id in range(0, len(test_set.get_all_sequences())):

        seq, length = test_set.get_item_Xlengths(word_id)
        words_probabilities = {}

        for word, model in models.items():
            try:
                words_probabilities[word] = model.score(seq, length)
            except:
                # taken from this url to pass the unittests: 
                # https://discussions.udacity.com/t/failure-in-recognizer-unit-tests/240082/5?u=cleyton_messias
                words_probabilities[word] = float('-inf')
                pass
        probabilities.append(words_probabilities)
        best_score = max(words_probabilities, key = words_probabilities.get)
        guesses.append(best_score)

    return probabilities, guesses
