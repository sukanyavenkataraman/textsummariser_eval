'''
Functions for the following 8 evaluations -
1. Non-redundancy
2. Precision, Recall, F1-score
3. Cosine similarity with tf-idf
4. Unit overlap/Jaccard similarity
5. Longest common subsequence
6. ROUGE
7. METEOR
8. Document categorisation
9. Information Retrieval
10. Cosine similarity with word2vec
'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from rogue import Rouge

def filter_chars(word):
    chars = ['\n', ' ', '\'', None]

    if word in chars:
        return False
    else:
        return True

def eval_non_redundancy(actual, predicted, phrase_len=4):
    '''
    Only checks for repeating phrases of length more than phrase_len
    :param actual: A list of actual summaries, index is the doc ID
    :param predicted: A list of predicted summaries, index is the doc ID
    :param phrase_len: Length of phrase to check for redundancy
    :return: List of documents with their actual, predicted redundancies
    '''


def eval_precision_recall_f1(actual, predicted):
    '''
    Precision: (sentences in actual, predicted/sentences in predicted)
    Recall: (sentences in actual, predicted/sentences in actual)
    F1-Score: 2PR/(P+R)
    :return: (Precision, Recall, F1-score)
    '''

    precision = recall = 0
    for i in range(len(actual)):

        #Ignore duplicate sentences, order
        actual_lines = set(actual[i].split('\n'))
        predicted_lines = set(predicted[i].split('\n'))

        intersection = actual_lines.intersection(predicted_lines)

        precision += 1.0*len(intersection)/len(predicted_lines)
        recall += 1.0*len(intersection)/len(actual_lines)

    precision = 1.0*precision/len(actual)
    recall = 1.0*recall/len(actual)
    f1_score = 2.0*precision*recall/(precision+recall)

    print ('Precision, recall, F1-score: ', precision, recall, f1_score)

    return (precision, recall, f1_score)

def eval_cosine_similarity(actual, predicted):
    '''
    cosine similarity based on tf
    Vectorise based on a vocab that is a combination of actual summary and predicted summary
    :return: float (avg. cosine similarity across all documents)
    '''

    cos_sim = 0

    for i in range(len(actual)):
        vocabulary = actual[i] + ' ' + predicted[i]

        vectoriser = CountVectorizer(input=vocabulary)
        vectoriser.fit([vocabulary])
        vectoriser._validate_vocabulary()

        actual_vector = vectoriser.transform([actual[i]])
        predicted_vector = vectoriser.transform([predicted[i]])

        cos_sim += cosine_similarity(actual_vector, predicted_vector)

    return 1.0*cos_sim/len(actual)

def eval_cosine_similarity_word2vec(actual, predicted):
    '''
    cosine similarity based on word2vec
    Vectorise based on a vocab that is a combination of actual summary and predicted summary
    :return: float (avg. cosine similarity across all documents)
    '''

    cosine_similarity = 0
    for i in range(len(actual)):
        '''
        vectoriser = Word2Vec()
        actual_vector = vectoriser.transform(actual[i])
        predicted_vector = vectoriser.transform(predicted[i])

        cosine_similarity += cosine_similarity(actual_vector, predicted_vector)
        '''

    return (1.0 * cosine_similarity / len(actual))

def eval_jaccard_distance(actual, predicted):

    '''
    Jaccard distance based on words
    :return: float (avg. jaccard distance across all documents)
    '''

    jacc_dist = 0
    for i in range(len(actual)):

        #Ignore duplicate words
        actual_words = set(filter(filter_chars, actual[i].split(' ')))
        predicted_words = set(filter(filter_chars, predicted[i].split(' ')))

        jacc_dist += nltk.jaccard_distance(actual_words, predicted_words)

    return (1.0 - 1.0*jacc_dist/len(actual))

def eval_longest_common_subsequence(actual, predicted):
    '''
    :return: avg. longest common subsequence between all documents
    '''

    lcs = 0
    for i in range(len(actual)):

        # Edit distance based on characters. TODO: Change to words?
        lcs += 1.0*(len(actual[i]) + len(predicted[i]) - nltk.edit_distance(actual[i], predicted[i]))/2

    return (1.0*(lcs)/len(actual))

def eval_rogue(actual, predicted):
    '''
    use the rogue library from (https://github.com/pltrdy/rouge) to calculate average
    rouge-1, rouge-2, rouge-L
    :return: avg. of scores - (rouge-1, rouge-2, rogue-L)
    '''

    r = Rouge()

    scores = r.get_scores(actual, predicted, avg=True)

    return scores

#def eval_meteor(actual, predicted):







