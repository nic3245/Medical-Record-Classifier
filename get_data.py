import pandas as pd
import os
import xml.etree.ElementTree as ET
def get_note_data(labels, folder_name='train', separate=False):
    if separate:
        headers = ["note1", "note2", "note3", "note4", "note5"]
        headers.extend(labels)
        overall_df = pd.DataFrame(columns=headers)
    else:
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
                    if separate:
                        notes = child.text.split("****************************************************************************************************")
                        notes = [note.strip() for note in notes if note.strip()]
                        i = 1
                        for note in notes:
                            row_to_add[f"note{i}"] = note
                            i += 1
                        for j in range(i, 6):
                            row_to_add[f"note{j}"] = ""
                    else:
                        note = child.text
                        row_to_add['notes'] = note
                if child.tag == "TAGS":
                    for subchild in child:
                        row_to_add[subchild.tag] = 1 if subchild.attrib.get('met') == 'met' else 0
            overall_df.loc[patient_num] = row_to_add

    return overall_df
