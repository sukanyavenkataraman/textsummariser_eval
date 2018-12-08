# Parse data depending on the data type into (document, summary)

import glob
import json

# CNN/daily-mail parser
def parse_cnn_dm(pathname):

    files = glob.glob(pathname+'*')

    documents = []
    summaries = []

    print (len(files))

    for file in files:
        next_highlight = False
        with open(file, 'r') as f:
            all_lines = f.readlines()
            story = []
            summary = []
            for line in all_lines:
                line = line.strip()
                line = line.replace('\n', '').replace('\r','')
                if line == '@highlight':
                    next_highlight = True
                    continue

                elif next_highlight is True:
                    if line == '':
                        continue
                    else:
                        summary.append(line)
                        next_highlight = False

                else:
                    story.append(line)
                    next_highlight = False

        story = list(filter(None, story))
        summary = list(filter(None, summary))

        documents.append('\n'.join(story))
        summaries.append('\n'.join(summary))

    print (len(documents), len(summaries))
    return documents, summaries

# Newsroom parser
def parse_newsroom(filename):
    documents = []
    summaries = []

    with open(filename, 'r') as f:
        all_lines = f.readlines()

        for line in all_lines:
            all_data = json.loads(line)

            documents.append(all_data['text'])
            summaries.append(all_data['summary'])

    print (len(documents), len(summaries))

    return documents, summaries


def parse_summary(pathname):

    files = glob.glob(pathname + '*')

    documents = []
    actual_summaries = []
    predicted_summaries = []

    print(len(files))

    for file in files:
        next_highlight_actual = False
        next_highlight_predicted = False

        with open(file, 'r') as f:
            all_lines = f.readlines()
            story = []
            actual_summary = []
            predicted_summary = []

            for line in all_lines:
                line = line.strip()
                if line == '@Actual_summary':
                    next_highlight_actual = True
                    continue

                elif line == '@Predicted_summary':
                    next_highlight_predicted = True
                    next_highlight_actual = False
                    continue

                elif next_highlight_actual is True:
                    if line == '':
                        continue
                    else:
                        actual_summary.append(line)

                elif next_highlight_predicted is True:
                    if line == '':
                        continue
                    else:
                        predicted_summary.append(line)

                else:
                    story.append(line)


        story = list(filter(None, story))
        actual_summary = list(filter(None, actual_summary))
        predicted_summary = list(filter(None, predicted_summary))

        documents.append('\n'.join(story))
        actual_summaries.append('\n'.join(actual_summary))
        predicted_summaries.append('\n'.join(predicted_summary))

    print(len(documents), len(actual_summaries), len(predicted_summaries))
    return documents, actual_summaries, predicted_summaries


# Newsroom summary parser
def parse_newsroom_summary():
    #TODO later
    return



