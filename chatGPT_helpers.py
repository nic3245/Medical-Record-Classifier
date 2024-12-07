# import openai
# from openai import OpenAI
import pickle
from utils import tokenize_data, truncate_clinical_note
import os

# Set up openAI api key
# key = os.environ.get("OPENAI_API_KEY")
# client = OpenAI(
#     api_key= key
# )

# openai.api_key = key


def zero_shot_prompt(note, model, tokenizer, label, MAX_TOKENS=3500, verbose=False):
    '''
    Asks the api if the given note meets the requirements associated with the given label.

    Arguments:
    note - the note to ask about
    label - the label whose requirements need to be met
    MAX_TOKENS - the maximum number of tokens to shorten the prompt to
    verbose - flag for extra prints
    '''
    requirements = {
    "DRUG-ABUSE": "Drug abuse, current or past",
    "ALCOHOL-ABUSE": "Current alcohol use over weekly recommended limits",
    "ENGLISH": "Patient must speak English",
    "MAKES-DECISIONS": "Patient must make their own medical decisions",
    "ABDOMINAL": "History of intra-abdominal surgery, small or large intestine resection, or small bowel obstruction.",
    "MAJOR-DIABETES": "Major diabetes-related complication. For the purposes of this annotation, we define “major complication” (as opposed to “minor complication”) as any of the following that are a result of (or strongly correlated with) uncontrolled diabetes: a. Amputation b. Kidney damage c. Skin conditions d. Retinopathy e. nephropathy f. neuropathy",
    "ADVANCED-CAD": "Advanced cardiovascular disease (CAD). For the purposes of this annotation, we define “advanced” as having 2 or more of the following: a. Taking 2 or more medications to treat CAD b. History of myocardial infarction (MI) c. Currently experiencing angina d. Ischemia, past or present",
    "MI-6MOS": "MI in the past 6 months",
    "KETO-1YR": "Diagnosis of ketoacidosis in the past year",
    "DIETSUPP-2MOS": "Taken a dietary supplement (excluding vitamin D) in the past 2 months",
    "ASP-FOR-MI": "Use of aspirin to prevent MI",
    "HBA1C": "Any hemoglobin A1c (HbA1c) value between 6.5% and 9.5%",
    "CREATININE": "Serum creatinine > upper limit of normal",
    }

    if verbose:
        print("predicting note:", note[10:15], "for label:", label) 
        print("truncating note...")

    # Truncate the note, if needed
    words = tokenize_data(note)
    prompt = f"Please check this requirement: f{requirements[label]}. Is this requirement met? Answer with YES, NO, or UNSURE."
    words_with_prompt = words + "\n" + prompt
    # Prompt the API
    inputs = tokenizer(words_with_prompt, return_tensors="pt").to("cuda")  # Send to GPU if available
    outputs = model.generate(inputs.input_ids, max_length=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    if verbose:
        print("Prediction complete with:", response)
    return response


def summarize_prompt(note, MAX_TOKENS=3500, verbose=False):
    '''
    Asks the api to summarize the given note to information about the label requirements.

    Arguments:
    note - the note to summarize
    MAX_TOKENS - the maximum number of tokens to shorten the prompt to
    verbose - flag for extra prints
    '''
    if verbose:
        print("truncating note...")
    # Shorten the note, if necessary
    words = truncate_clinical_note(tokenize_data(note), MAX_TOKENS)
    prompt = ("please summarize the clinical note above to the following information:\n"
            "1. DRUG-ABUSE: Drug abuse, current or past\n"
            "2. ALCOHOL-ABUSE: Current alcohol use over weekly recommended limits\n"
            "3. ENGLISH: Patient must speak English\n"
            "4. MAKES-DECISIONS: Patient must make their own medical decisions\n"
            "5. ABDOMINAL: History of intra-abdominal surgery, small or large intestine resection, or small bowel obstruction.\n"
            "6. MAJOR-DIABETES: Major diabetes-related complication. For the purposes of this annotation, we define 'major complication' (as opposed to 'minor complication') as any of the following that are a result of (or strongly correlated with) uncontrolled diabetes:\n"
            "   a. Amputation\n"
            "   b. Kidney damage\n"
            "   c. Skin conditions\n"
            "   d. Retinopathy\n"
            "   e. nephropathy\n"
            "   f. neuropathy\n"
            "7. ADVANCED-CAD: Advanced cardiovascular disease (CAD). For the purposes of this annotation, we define 'advanced' as having 2 or more of the following:\n"
            "   a. Taking 2 or more medications to treat CAD\n"
            "   b. History of myocardial infarction (MI)\n"
            "   c. Currently experiencing angina\n"
            "   d. Ischemia, past or present\n"
            "8. MI-6MOS: MI in the past 6 months\n"
            "9. KETO-1YR: Diagnosis of ketoacidosis in the past year\n"
            "10. DIETSUPP-2MOS: Taken a dietary supplement (excluding vitamin D) in the past 2 months\n"
            "11. ASP-FOR-MI: Use of aspirin to prevent MI\n"
            "12. HBA1C: Any hemoglobin A1c (HbA1c) value between 6.5% and 9.5%\n"
            "13. CREATININE: Serum creatinine > upper limit of normal")
    words_with_prompt = words + "\n" + prompt
    if verbose:
        print("Asking for summary...")
    # Prompt the api
    chat_completion = openai.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"{words_with_prompt}",
            }
        ],
        model="gpt-3.5-turbo",
        max_tokens=512,
        temperature=.3
    )
    # Get the response
    response = chat_completion.choices[0].message.content
    if verbose:
        print("Summarization complete")
    return response

def summarize_and_save_notes(notes, name='train', MAX_TOKENS=3500, verbose=False):
    '''
    Summarize the given list of notes wiht the api.

    Arguments:
    notes - the list of notes to summarize
    name - the name to save the summaries with
    MAX_TOKENS - the maximum number of tokens to shorten the prompt to
    verbose - flag for extra prints
    '''
    summaries = []
    # Get the summaries
    for i in range(len(notes)):
        if verbose:
            print("predicting note:", i)
        summaries.append(summarize_prompt(notes[i], verbose=verbose))
    if verbose:
        print("Number of summaries completed:", len(summaries))
    # Save the summaries
    file_path = f'summarized_notes_{name}.pkl'
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(summaries, file)

        # Check that it saved correctly (done b/c each run is $$$$)
        with open(file_path, 'rb') as file:
            loaded_list = pickle.load(file)

        # Compare the original list with the loaded list
        if summaries == loaded_list:
            print("List was saved and loaded correctly.")
        else:
            print("There was an issue with saving or loading the list.")
        return summaries
    except RuntimeError:
        return summaries