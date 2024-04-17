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
    save - Flag to save the model
    save_name - What the directory for the model should be named

    Returns:
    TFAutoModelForSequenceClassification
    '''
    # Load tokenizer and model
    if return_history and (test_data is None):
        raise ValueError("Must give test_data to get return_history")
    
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model_ = TFAutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model_.trainable = False
    # setting trainable to false ensures
    # we do not update its weights
    model = tf.keras.Sequential([
        model_,
        tf.keras.layers.Lambda(lambda x: x[0][:,0,:]), # https://keras.io/api/layers/core_layers/lambda/
        tf.keras.layers.Dense(512,activation="relu"),
        tf.keras.layers.Dense(512,activation="relu"),
        tf.keras.layers.Dense(1,activation="sigmoid")
    ])

    clinical_notes = [truncate_clinical_note(tokenize_data(note), 512) for note in list(df['notes'])]
    # Tokenize and pad the clinical notes
    tokenized_notes = tokenizer(clinical_notes, padding='max_length', max_length=512, truncation=True, return_tensors="tf")
    tokenized_data = dict(tokenized_notes)


    labels = np.array(df[label])

    if return_history:
        cn_test = [truncate_clinical_note(tokenize_data(note), 512) for note in list(test_data['notes'])]
        tokenized_test = dict(tokenizer(cn_test, padding='max_length', max_length=512, truncation=True, return_tensors="tf"))
        labels_test = np.array(test_data[label])

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=.1, patience=2, monitor='loss')
    model.compile(loss="binary_crossentropy", optimizer=Adam(0.001),metrics=['accuracy'])

    if return_history:
        history = model.fit(tokenized_data, labels, epochs=num_epochs, verbose=True, callbacks=[lr_scheduler],
                            class_weight ={0: np.size(labels) / np.sum(labels == 0), 1: np.size(labels) / np.sum(labels == 1)},
                            validation_data=(tokenized_test, labels_test))
    else:
        model.fit(tokenized_data, labels, epochs=num_epochs, verbose=True, callbacks=[lr_scheduler],
                            class_weight ={0: np.size(labels) / np.sum(labels == 0), 1: np.size(labels) / np.sum(labels == 1)})

    if return_history:
        return model, history
    else:
        return model

def for_passed_labels(labels, df, epochs=10, return_history=False, verbose=False, test_data=None):
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
    # Tokenize and pad the clinical notes
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    clinical_notes = [truncate_clinical_note(tokenize_data(note), 512) for note in list(test_data['notes'])]
    tokenized_notes = tokenizer(clinical_notes, padding='max_length', max_length=512, truncation=True, return_tensors="tf")
    tokenized_data = dict(tokenized_notes)
    # Get preds
    model_predictions = model.predict(tokenized_data)
    try:
        probabilities = tf.nn.sigmoid(model_predictions.logits).numpy()
        predictions = np.where(probabilities < threshold, 0, 1)
        return predictions
    except AttributeError:
        probabilities = tf.nn.sigmoid(model_predictions).numpy()
        predictions = np.where(probabilities < threshold, 0, 1)[:,0].tolist()
        return predictions


def create_multi_label(train_data, test_data=None, num_epochs=10, save=True):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = TFAutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=13, problem_type="multi_label_classification")
    model.layers[0].trainable = False

    # Tokenize and pad the clinical notes
    clinical_notes = list(train_data['notes'])
    clinical_notes = [truncate_clinical_note(tokenize_data(note)) for note in clinical_notes]
    tokenized_notes = tokenizer(clinical_notes, padding='max_length', max_length=512, truncation=True, return_tensors="tf")
    tokenized_data = dict(tokenized_notes)
    labels = np.array(train_data.drop(columns=['notes'], inplace=False))

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

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=.5, patience=3, monitor='loss')
    model.compile(loss="binary_crossentropy", optimizer=Adam(0.001),metrics=['accuracy']) 

    if not test_data is None:
        history = model.fit(tokenized_data, labels, epochs=num_epochs, verbose=True, callbacks=[lr_scheduler],
                             class_weight=class_weights_dict, validation_data=(tokenized_test, labels_test))
    else:
        history = model.fit(tokenized_data, labels, epochs=num_epochs, verbose=True, callbacks=[lr_scheduler],
                             class_weight=class_weights_dict)

    if save:
        model.save_pretrained('overall_model')

    return model, history
