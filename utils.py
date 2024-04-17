import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import re

def tokenize_data(note):
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    pattern = r"[\n\s*]+|\.\s+"
    # Split the text into sentences
    word_tokens = re.split(pattern, note.lower())
    # Remove stop words
    filtered_tokens = [token for token in word_tokens if token not in stop_words and token != '']
    return " ".join(filtered_tokens)


def truncate_clinical_note(note, MAX_TOKENS = 3500):
    tokenized_words = tokenize_data(note).split()
    num_tokens = len(tokenized_words)

    if num_tokens < MAX_TOKENS:
        return " ".join(tokenized_words)

    truncation_step = (num_tokens - MAX_TOKENS) // MAX_TOKENS + 1
    
    tokenized_words = tokenized_words[::truncation_step]
    return " ".join(tokenized_words)

def display_histogram(data):
    lengths = [len(note.split()) for note in data['notes']]

    x = [length for length in lengths]

    plt.hist(x,bins=30)
    plt.ylabel("Count")
    plt.xlabel("note Length")
    plt.title("Lengths of clinical notes")
    plt.show()

def get_note_data(labels, folder_name='train'):
    headers = ["notes"]
    headers.extend(labels)
    overall_df = pd.DataFrame(columns=headers)

    current_directory = os.getcwd()
    directory = os.path.join(current_directory, folder_name)
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            patient_num = os.path.splitext(filename)[0]
            row_to_add = {}
            # Load the XML file
            tree = ET.parse(os.path.join(directory, filename))
            root = tree.getroot()
            # Access elements and attributes
            for child in root:
                if child.tag == "TEXT":
                    note = child.text
                    row_to_add['notes'] = note
                if child.tag == "TAGS":
                    for subchild in child:
                        row_to_add[subchild.tag] = 1 if subchild.attrib.get('met') == 'met' else 0
            overall_df.loc[patient_num] = row_to_add

    return overall_df


def save_preds(label_to_predictions, name):
    dfp = pd.DataFrame.from_dict(label_to_predictions, orient='index')
    dfp.to_csv(f'{name}.csv')

def read_preds(name):
    # Read in prediction data:
    dfp = pd.read_csv(f'{name}.csv')
    label_to_predictions = {}
    for index, row in dfp.iterrows():
        label_to_predictions[row[0]] = list(row[1:])
    return label_to_predictions

def get_f1_scores_for_labels(labels, y_true_source, y_pred_source, verbose=True):
    label_to_micro_f1 = {}
    for label in labels:
        if label in y_pred_source:
            mtp, mtn, mfp, mfn = 0, 0, 0, 0
            ntp, ntn, nfp, nfn = 0, 0, 0, 0
            for true, pred in zip(y_true_source[label], y_pred_source[label]):
                if true == 1:
                    if pred == true:
                        mtp += 1
                        ntn += 1
                    else:
                        mfn += 1
                        nfp += 1
                else:
                    if pred == true:
                        mtn += 1
                        ntp += 1
                    else:
                        mfn += 1
                        mfp += 1
            if mtp == 0:
                mprecision = 0
                mrecall = 0
            else:
                mprecision = mtp / (mtp + mfp)
                mrecall = mtp / (mtp + mfn)
            if ntp == 0:
                nprecision = 0
                nrecall = 0
            else:
                nprecision = ntp / (ntp + nfp)
                nrecall = ntp / (ntp + nfn)
            if mprecision == 0 or mrecall == 0:
                mf1 = 0
            else:
                mf1 = 2*mprecision*mrecall / (mprecision + mrecall)
            if nprecision == 0 or nrecall == 0:
                nf1 = 0
            else:
                nf1 = 2*nprecision*nrecall / (nprecision + nrecall)
            micro_f1 = (mf1 + nf1) / 2
            label_to_micro_f1[label] = micro_f1
            if verbose:
                print("Raw f1 for", label, micro_f1)
    overall_f1 = np.mean(list(label_to_micro_f1.values()))
    return label_to_micro_f1, overall_f1

def make_history_graph(history):
    # history for accuracy
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()
    # history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()