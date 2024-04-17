import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel, TFAutoModelForSequenceClassification
import numpy as np
from tensorflow.keras.optimizers import Adam
from utils import tokenize_data, truncate_clinical_note

def for_one_label(df, label, num_epochs=10, return_history=False, test_data=None):
    '''
    Makes a fine-tuned ClinicalBERT model for a given label with data.

    Arguments:
    df - DataFrame containing a 'notes' column with the corresponding clinical notes
    label - Str that is the name of the column to use as y/labels for the model training
    return_history - flag to return the history of the training
    test_data - DataFrame containing a 'notes' column with the corresponding test data notes

    If return_history=True, test_data must be given.

    Returns:
    if return_history: TFAutoModelForSequenceClassification, tf.keras.callback.History
    else: TFAutoModelForSequenceClassification
    '''
    # Check test data
    if return_history and (test_data is None):
        raise ValueError("Must give test_data to get return_history")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model_ = TFAutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model_.trainable = False  # setting trainable to false ensures we do not update its weights

    # Set up fine-tuning FFNN
    model = tf.keras.Sequential([
        model_,
        tf.keras.layers.Lambda(lambda x: x[0][:,0,:]), # https://keras.io/api/layers/core_layers/lambda/
        tf.keras.layers.Dense(512,activation="relu"),
        tf.keras.layers.Dense(512,activation="relu"),
        tf.keras.layers.Dense(1,activation="sigmoid")
    ])
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=.1, patience=2, monitor='loss')
    model.compile(loss="binary_crossentropy", optimizer=Adam(0.001),metrics=['accuracy'])

    # Set up training data
    clinical_notes = [truncate_clinical_note(tokenize_data(note), 512) for note in list(df['notes'])]
    tokenized_notes = tokenizer(clinical_notes, padding='max_length', max_length=512, truncation=True, return_tensors="tf")
    tokenized_data = dict(tokenized_notes)
    labels = np.array(df[label])
    # Set up test data if needed
    if return_history:
        cn_test = [truncate_clinical_note(tokenize_data(note), 512) for note in list(test_data['notes'])]
        tokenized_test = dict(tokenizer(cn_test, padding='max_length', max_length=512, truncation=True, return_tensors="tf"))
        labels_test = np.array(test_data[label])
    
    # Train model
    if return_history:
        history = model.fit(tokenized_data, labels, epochs=num_epochs, verbose=True, callbacks=[lr_scheduler],
                            class_weight ={0: np.size(labels) / np.sum(labels == 0), 1: np.size(labels) / np.sum(labels == 1)},
                            validation_data=(tokenized_test, labels_test))
    else:
        model.fit(tokenized_data, labels, epochs=num_epochs, verbose=True, callbacks=[lr_scheduler],
                            class_weight ={0: np.size(labels) / np.sum(labels == 0), 1: np.size(labels) / np.sum(labels == 1)})

    # Return results
    if return_history:
        return model, history
    else:
        return model

def for_passed_labels(labels, df, epochs=10, return_history=False, test_data=None, verbose=False):
    '''
    Makes a model for each label using the given data and labels.

    Arguments:
    labels - List[Str] where each string is a label
    df - DataFrame of the training data (has a notes column and a column for each label)
    return_history - flag to return the history of the training
    test_data - DataFrame containing a 'notes' column with the corresponding test data notes
    verbose - flag for extra prints

    Returns:
    Dict Str -> TFAutoModelForSequenceClassification
    '''
    models = {}
    histories = {}
    for label in labels:
        if verbose:
            print(f"Making model for label {label}...")
        if return_history:
            model, history = for_one_label(df, label, num_epochs=epochs, return_history=return_history, test_data=test_data)
        else:
            model = for_one_label(df, label, num_epochs=epochs)
        if verbose:
            print("Finished making model.")
        models[label] = model
        if return_history:
            histories[label] = history
    if return_history:
        return models, histories
    else:
        return models

def get_predictions(model, test_data, threshold=.5, verbose=True):
    '''
    Gets the predictions of the test_data using the given model.

    Arguments:
    model - model to use for the predictions
    test_data - DataFrame with a 'notes' column to use as input
    threshold - probabilities >= will be 1, < will be 0
    verbose - flag for extra prints

    Returns:
    np.Array() (multi-label model) or List[] (single-label model) representing predictions.
    '''
    # Prepare Data
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    clinical_notes = [truncate_clinical_note(tokenize_data(note), 512) for note in list(test_data['notes'])]
    tokenized_notes = tokenizer(clinical_notes, padding='max_length', max_length=512, truncation=True, return_tensors="tf")
    tokenized_data = dict(tokenized_notes)
    # Get Predictions
    model_predictions = model.predict(tokenized_data)
    try: # If it stays as a HuggingFace model, it will return logits
        probabilities = tf.nn.sigmoid(model_predictions.logits).numpy()
        predictions = np.where(probabilities < threshold, 0, 1)
        return predictions
    except AttributeError: # If it doesn't, it will just be a numpy array
        probabilities = tf.nn.sigmoid(model_predictions).numpy()
        predictions = np.where(probabilities < threshold, 0, 1)[:,0].tolist()
        return predictions


def create_multi_label(train_data, test_data=None, num_epochs=10, save=True):
    '''
    Makes a fine-tuned mulit-label ClinicalBERT model for the given train data.

    Arguments:
    train_data - DataFrame with a 'notes' column to be used as X input and 13 other columns as y input
    test_data - Same as train_data, but for validation
    num_epochs - number of epochs to train the model for
    save - flag to save the model

    Returns:
    TFAutoModelForSequenceClassification, tf.Keras.Callback.History
    '''
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = TFAutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=13, problem_type="multi_label_classification")
    model.layers[0].trainable = False
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=.5, patience=3, monitor='loss')
    model.compile(loss="binary_crossentropy", optimizer=Adam(0.001),metrics=['accuracy']) 

    # Prepare train data
    clinical_notes = list(train_data['notes'])
    clinical_notes = [truncate_clinical_note(tokenize_data(note)) for note in clinical_notes]
    tokenized_notes = tokenizer(clinical_notes, padding='max_length', max_length=512, truncation=True, return_tensors="tf")
    tokenized_data = dict(tokenized_notes)
    labels = np.array(train_data.drop(columns=['notes'], inplace=False))
    # Prepare test data
    if not test_data is None:
        cn_test = [truncate_clinical_note(tokenize_data(note), 512) for note in list(test_data['notes'])]
        tokenized_test = dict(tokenizer(cn_test, padding='max_length', max_length=512, truncation=True, return_tensors="tf"))
        labels_test = np.array(test_data.drop(columns=['notes'], inplace=False))

    # Set up class weights
    class_weights_dict = {}
    for i in range(labels.shape[1]):
        class_counts = np.sum(labels[:, i])  # Count of positive samples for current class
        total_samples = len(labels)
        class_weight = (total_samples - class_counts) / class_counts  # Inverse of class frequency
        class_weights_dict[i] = class_weight

    # Train model
    if not test_data is None:
        history = model.fit(tokenized_data, labels, epochs=num_epochs, verbose=True, callbacks=[lr_scheduler],
                             class_weight=class_weights_dict, validation_data=(tokenized_test, labels_test))
    else:
        history = model.fit(tokenized_data, labels, epochs=num_epochs, verbose=True, callbacks=[lr_scheduler],
                             class_weight=class_weights_dict)

    if save:
        model.save_pretrained('overall_model')

    return model, history
