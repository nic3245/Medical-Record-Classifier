import openai
from openai import OpenAI
import pickle
from utils import tokenize_data, truncate_clinical_note


key = "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"
client = OpenAI(
    # This is the default and can be omitted
    #api_key=os.environ.get("OPENAI_API_KEY"),
    api_key= key
)

openai.api_key = key


def zero_shot_prompt(note, label, MAX_TOKENS=3500):
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
    print("predicting note:", note[10:15], "for label:", label)
    print("truncating note...")
    words = truncate_clinical_note(tokenize_data(note), MAX_TOKENS)
    prompt = f"Please check this requirement: f{requirements[label]}. Is this requirement met? Answer with YES, NO, or UNSURE."
    words_with_prompt = words + "\n" + prompt

    chat_completion = openai.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"{words_with_prompt}",
            }
        ],
        model="gpt-3.5-turbo",
        max_tokens=2,
        temperature=.3
    )

    response = chat_completion.choices[0].message.content
    print("Prediction complete with:", response)
    return response


def summarize_prompt(note, MAX_TOKENS=3500):
    print("truncating note...")
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
    print("Asking for summary...")
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

    response = chat_completion.choices[0].message.content
    print("Summarization complete")
    return response

def summarize_and_save_notes(notes, name='train', MAX_TOKENS=3500):
    summaries = []
    for i in range(len(notes)):
        print("predicting note:", i)
        summaries.append(summarize_prompt(notes[i]))
    print("Number of summaries completed:", len(summaries))
    # Specify the file path
    file_path = f'summarized_notes_{name}.pkl'

    try:
        # Write the list to the file using pickle
        with open(file_path, 'wb') as file:
            pickle.dump(summaries, file)

        # Load the list back from the file
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
    





note = "Past medical history: 1. CAD- details above 2. DM - on glyburide and metformin 3. Trigeminal neuralgia - on lyrica and tegretol 4. Hypothyroidism - on synthroid 5. Hyperlipidemia 6. HTN"
note_2 = "He was found to be in congestive heart failure. He was given plavix, aspirin, lovenox and IV lasix with improvement in symptoms. On 08/31/2135, he was transferred to Gibson Community Hospital for cardiac catheterization. The results were significant for multivessel coronary artery disease and moderate mitral valve regurgitation with an left ventricular ejection fraction of 25 to 30%. He was referred to Dr. Sebastian Dang for elective cardiac surgical revascularization and possible mitral valve repair. He was discharged to home on 09/02/2135."
# clinal_note_1 = patient_notes_data['notes']
# assert generate_text_from_label(clinal_note_1, label='ABDOMINAL')


# Example test of summarizing
# print(summarize_prompt(note_2))

