from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .preprocess import clean_text

def get_similarity(resume_text, jd_text):
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([jd_clean, resume_clean])

    # Cosine Similarity
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    # Feature names (vocabulary)
    feature_names = vectorizer.get_feature_names_out()

    # Vectors for JD and Resume
    jd_vector = vectors[0].toarray()[0]
    resume_vector = vectors[1].toarray()[0]

    # Matched keywords: present in both
    matched_keywords = []
    for i in range(len(feature_names)):
        if jd_vector[i] > 0 and resume_vector[i] > 0:
            matched_keywords.append((feature_names[i], jd_vector[i]))

    matched_keywords.sort(key=lambda x: x[1], reverse=True)
    top_keywords = [word for word, score in matched_keywords[:10]]

    # 🔥 New: Skill gaps (JD keywords not in resume)
    all_jd_keywords = set([feature_names[i] for i in range(len(feature_names)) if jd_vector[i] > 0])
    all_resume_keywords = set([feature_names[i] for i in range(len(feature_names)) if resume_vector[i] > 0])
    gap_keywords = list(all_jd_keywords - all_resume_keywords)
    gap_keywords = sorted(gap_keywords)[:10]  # limit to top 10

    return round(similarity * 100, 2), top_keywords, gap_keywords






from sentence_transformers import SentenceTransformer, util

# Load model once globally (you can move this outside function if needed)
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_semantic_similarity(resume_text, jd_text):
    # Clean (optional: can skip heavy cleaning for BERT)
    resume = resume_text.strip()
    jd = jd_text.strip()

    # Encode texts into embeddings
    resume_embedding = bert_model.encode(resume, convert_to_tensor=True)
    jd_embedding = bert_model.encode(jd, convert_to_tensor=True)

    # Cosine similarity
    similarity = util.pytorch_cos_sim(resume_embedding, jd_embedding).item()

    # Convert to percentage
    match_percent = round(similarity * 100, 2)

    return match_percent

