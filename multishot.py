def tokenize_data(note):
    import nltk
    from nltk.corpus import stopwords
    import re
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    pattern = r"[\n\.]"
    # Split the text into sentences
    sentences = re.split(pattern, note.lower())
    # Split the text into tokens
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences if sentence.strip()]
    # Remove stop words
    #filtered_tokens = [token for token in tokens if token not in stop_words]
    return tokens