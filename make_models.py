import torch
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np
from tensorflow.keras.optimizers import Adam

def for_one_label(df, label, num_epochs, save=False, save_name=None):
    '''
    Makes a fine-tuned ClinicalBERT model for a given label with data.

    Arguments:
    df - DataFrame containing a 'notes' column with the corresponding clinical notes
    label - Str that is the name of the column to use as y/labels for the model training
    save - Flag to save the model
    save_name - What the directory for the model should be named

    Returns:
    TFAutoModelForSequenceClassification
    '''
    if save and save_name is None:
        save_name = f'{label}_model'
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = TFAutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", from_pt=True)
    clinical_notes = list(df['notes'])

    # Tokenize and pad the clinical notes
    tokenized_notes = tokenizer(clinical_notes, padding='max_length', max_length=512, truncation=True, return_tensors="tf")
    tokenized_data = dict(tokenized_notes)
    labels = np.array(df[label])

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=.5, patience=3, monitor='loss')
    model.compile(optimizer=Adam(3e-5), metrics='accuracy')

    model.fit(tokenized_data, labels, epochs=num_epochs, verbose=True, callbacks=[lr_scheduler],class_weight ={0: np.size(labels) / np.sum(labels == 0), 1: np.size(labels) / np.sum(labels == 1)})
    if save:
        tokenizer.save_pretrained(save_name)
        model.save_pretrained(save_name)
    return model

def for_passed_labels(labels, make_model_function, df, epochs, save=False, verbose=False):
    '''
    Makes a model for each label using the given function, data, and labels.

    Arguments:
    labels - List[Str] where each string is a label
    make_model_function - Callable to make the model that takes in a df and label
    df - DataFrame of the data
    save - Flag to save each model

    Returns:
    Dict Str -> TFAutoModelForSequenceClassification
    '''
    models = {}
    for label in labels:
        if verbose:
            print(f"Making model for label {label}...")
        model = make_model_function(df, label, epochs, save=save)
        if verbose:
            print("Finished making model.")
        models[label] = model
    return models