import streamlit as st
import pandas as pd
import PyPDF2
from utils.matcher import get_similarity, get_semantic_similarity

st.title("Resume Matcher")

# Choose matching type
match_type = st.radio("Choose Matching Method", ["TF-IDF", "BERT (Semantic)"])

# Job description input
jd_text = st.text_area("Paste Job Description")

# Resume upload
uploaded_files = st.file_uploader(
    "Upload One or More Resumes (.pdf or .txt)", 
    type=['pdf', 'txt'], 
    accept_multiple_files=True
)

# Match resumes
if st.button("Match Resumes"):
    if uploaded_files and jd_text:
        results = []

        for file in uploaded_files:
            try:
                # Extract text
                if file.name.endswith('.pdf'):
                    reader = PyPDF2.PdfReader(file)
                    resume_text = ""
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            resume_text += text
                else:
                    resume_text = file.read().decode()

                # Choose method
                if match_type == "TF-IDF":
                    score, keywords, gaps = get_similarity(resume_text, jd_text)
                else:
                    score = get_semantic_similarity(resume_text, jd_text)
                    keywords, gaps = [], []

                results.append((file.name, score, keywords, gaps))

            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {e}")

        # Sort results by score
        results.sort(key=lambda x: x[1], reverse=True)

        # Display results
        st.subheader("Ranked Resume Matches")
        for name, score, keywords, gaps in results:
            st.markdown(f"**📄 {name}** — Match: `{score}%` via `{match_type}`")
            if match_type == "TF-IDF":
                st.markdown(f"Matched Keywords: _{', '.join(keywords)}_")
                st.markdown(f"Missing Key Terms: _{', '.join(gaps)}_")
            st.markdown("---")

        # Create DataFrame
        df = pd.DataFrame(results, columns=["Resume", "Match %", "Matched Keywords", "Missing Keywords"])
        df["Matched Keywords"] = df["Matched Keywords"].apply(lambda x: ", ".join(x))
        df["Missing Keywords"] = df["Missing Keywords"].apply(lambda x: ", ".join(x))

        # Bar Chart
        st.subheader("Resume Match Score Chart")
        st.bar_chart(df.set_index("Resume")["Match %"])

        # Table
        st.subheader("Resume Match Table")
        st.dataframe(df)

    else:
        st.warning("⚠️ Please upload at least one resume and paste the job description.")
