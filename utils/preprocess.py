# import re
# import nltk
# from nltk.corpus import stopwords

# nltk.download('stopwords')
# STOPWORDS = set(stopwords.words('english'))

# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'\n', ' ', text)
#     text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
#     text = ' '.join([word for word in text.split() if word not in STOPWORDS])
#     return text



import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text(text):
    # Lowercase everything
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]

    return ' '.join(words)


#I built a preprocessing function that converts the text to lowercase, removes numbers, punctuation, extra spaces, and stopwords. This helps reduce noise in the text and makes semantic comparison more meaningful for models like TF-IDF or SBERT.
