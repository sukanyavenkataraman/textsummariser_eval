from topicModel import TopicModel
from documentSummaries import DocumentSummaries
from datagenerator import parse_cnn_dm, parse_newsroom, parse_summary
from evaluation import eval_cosine_similarity, eval_cosine_similarity_word2vec, eval_jaccard_distance, eval_longest_common_subsequence, eval_precision_recall_f1, eval_rogue, eval_non_redundancy

import time

def summarise_docs(num_topics=100, dataset_type='cnn'):

    start_time = time.time()

    if dataset_type == 'cnn':
        datadir = '/Users/sukanya/Downloads/cnn/stories/'#_stories_tokenized/'
        documents, summaries = parse_cnn_dm(datadir)

    elif dataset_type == 'dailymail':
        datadir = '/Users/sukanya/Downloads/dailymail/stories/'#_stories_tokenized/'
        documents, summaries = parse_cnn_dm(datadir)

    elif dataset_type == 'newsroom':
        datadir = '/Users/sukanya/Downloads/newsroom/test.jsonl'
        documents, summaries = parse_newsroom(datadir)

    else:
        #TODO:Remove this
        print ('Dataset type unsupported')
        return

    num_dominant_topics = 5
    num_sentences = 4

    parse_docs_time = time.time()
    print ('Time taken to parse documents: ', parse_docs_time - start_time)

    print (len(documents))#, comments)
    topicModel = TopicModel(num_topics)
    topicModel.fit(documents, num_dominant_topics)

    topic_model_time = time.time()
    print('Time taken to model topics: ', topic_model_time - parse_docs_time)

    for index, document in enumerate(documents):
        docSummaries = DocumentSummaries(topicModel, num_dominant_topics=num_dominant_topics, number_of_sentences=num_sentences)
        docSummaries.summarize(document)
        print (index)
        summary = docSummaries.summary()

        f = open('/Users/sukanya/Downloads/'+dataset_type+'/stories_with_summary/document_' + str(index) + '.txt', 'w+')

        f.write(document + '\n@Actual_summary\n' + summaries[index] + '\n@Predicted_summary\n' + summary)
        f.close()
        #docSummaries.display()

    gen_summary_time = time.time()
    print('Time taken to generate summaries: ', gen_summary_time - topic_model_time)
    print('Total time taken: ', gen_summary_time - start_time)


def evaluate_summaries(dataset_type='cnn'):
    start_time = time.time()

    if dataset_type == 'cnn':
        datadir = '/Users/sukanya/Downloads/cnn/stories_with_summary/'  # _stories_tokenized/'
        documents, actual_summaries, predicted_summaries = parse_summary(datadir)

    elif dataset_type == 'dailymail':
        datadir = '/Users/sukanya/Downloads/dailymail/stories_with_summary/'  # _stories_tokenized/'
        documents, actual_summaries, predicted_summaries = parse_summary(datadir)

    elif dataset_type == 'newsroom':
        datadir = '/Users/sukanya/Downloads/newsroom/stories_with_summary/'
        documents, actual_summaries, predicted_summaries = parse_summary(datadir)

    else:
        # TODO:Remove this
        print('Dataset type unsupported')
        return

    document_parsing_time = time.time()
    print ('Time to parse docs: ', document_parsing_time-start_time)

    # Precision, recall, f1
    precision, recall, f1_score = eval_precision_recall_f1(actual_summaries, predicted_summaries)
    print('Precision: %f, Recall: %f, F1-score: %f' %(precision, recall, f1_score))

    #Cosine similarity based on tf
    cosine_similarity = eval_cosine_similarity(actual_summaries, predicted_summaries)
    print('Cosine similarity: ', cosine_similarity)

    #Jaccard similarity
    jacc_sim = eval_jaccard_distance(actual_summaries, predicted_summaries)
    print('Jaccard similarity: ', jacc_sim)

    #LCS
    #lcs = eval_longest_common_subsequence(actual_summaries, predicted_summaries)
    #print('Average longest common subsequence: ', lcs)

    #Rogue
    rogue_scores = eval_rogue(actual_summaries, predicted_summaries)
    print('Rouge-1: ', rogue_scores['rouge-1'])
    print('Rouge-2: ', rogue_scores['rouge-2'])
    print('Rouge-L: ', rogue_scores['rouge-l'])

    print ('Time to eval: ', time.time() - document_parsing_time)

summarise = False

if summarise:
    summarise_docs(dataset_type='newsroom')
else:
    evaluate_summaries(dataset_type='newsroom')


