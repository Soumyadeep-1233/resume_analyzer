import streamlit as st
import fitz  # PyMuPDF
from docx import Document
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from unidecode import unidecode
import os
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Resume Analyzer | Job Match Score",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Download spaCy model if not already installed
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

# Sample job descriptions (replace with your actual data)
JOB_DESCRIPTIONS = {
    "Data Scientist": """
        Responsibilities:
        - Analyze large datasets
        - Build machine learning models
        - Communicate insights
        Requirements:
        - Python, SQL
        - TensorFlow/PyTorch
        - Statistical analysis
        - Data visualization
    """,
    "Software Engineer": """
        Responsibilities:
        - Develop scalable applications
        - Write clean, maintainable code
        - Collaborate with teams
        Requirements:
        - Java/Python/JavaScript
        - Algorithms & data structures
        - Software development lifecycle
    """,
    "Project Manager": """
        Responsibilities:
        - Lead project execution
        - Manage stakeholders
        - Ensure on-time delivery
        Requirements:
        - Agile methodologies
        - Risk management
        - Communication skills
        - Budgeting
    """
}

# Preprocessing functions
def clean_text(text):
    """Normalize text"""
    text = unidecode(text)  # Convert accented characters
    text = re.sub(r'\W', ' ', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text.lower()

def extract_skills(text):
    """Extract technical skills using spaCy"""
    doc = nlp(text)
    skills = set()
    
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            skills.add(ent.text.lower())
    
    # Manual skill extraction as fallback
    skill_keywords = [
        "python", "java", "sql", "machine learning", "aws",
        "docker", "kubernetes", "agile", "scrum", "javascript",
        "react", "tensorflow", "pytorch", "tableau", "excel"
    ]
    
    for skill in skill_keywords:
        if skill in text.lower():
            skills.add(skill)
    
    return list(skills)

def extract_text_from_file(uploaded_file):
    """Extract text from PDF/DOCX"""
    text = ""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type == "pdf":
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text = " ".join([page.get_text() for page in doc])
    elif file_type == "docx":
        doc = Document(BytesIO(uploaded_file.read()))
        text = " ".join([para.text for para in doc.paragraphs])
    
    return clean_text(text)

def calculate_match(resume_text, job_description):
    """Calculate cosine similarity between resume and JD"""
    vectorizer = CountVectorizer().fit_transform([resume_text, job_description])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0] * 100

def analyze_resume(resume_text, selected_job):
    """Core analysis function"""
    # Extract and process
    skills = extract_skills(resume_text)
    jd_text = clean_text(JOB_DESCRIPTIONS[selected_job])
    match_score = round(calculate_match(resume_text, jd_text), 1)
    
    # Get missing skills
    jd_skills = extract_skills(jd_text)
    missing_skills = [skill for skill in jd_skills if skill not in skills]
    
    return {
        "match_score": match_score,
        "skills": skills,
        "missing_skills": missing_skills,
        "job_description": JOB_DESCRIPTIONS[selected_job]
    }

# Streamlit UI
def main():
    st.title("📄 Resume Analyzer with Job Match Score")
    st.write("Upload your resume and see how well it matches your target job!")
    
    with st.sidebar:
        st.header("Settings")
        selected_job = st.selectbox("Select Job Role:", list(JOB_DESCRIPTIONS.keys()))
        uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX):", type=["pdf", "docx"])
    
    if uploaded_file and selected_job:
        try:
            # Process resume
            resume_text = extract_text_from_file(uploaded_file)
            analysis_results = analyze_resume(resume_text, selected_job)
            
            # Display results
            st.success("Analysis Complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Match Score", f"{analysis_results['match_score']}%")
            
            with col2:
                st.metric("Key Skills Found", len(analysis_results['skills']))
            
            st.divider()
            
            # Skills section
            st.subheader("🔍 Skill Analysis")
            st.write("**Your Skills:**")
            st.write(", ".join(analysis_results['skills']) if analysis_results['skills'] else "No skills detected")
            
            if analysis_results['missing_skills']:
                st.warning(f"**Skills to improve:** {', '.join(analysis_results['missing_skills'])}")
            
            st.divider()
            
            # Job Description preview
            with st.expander("📝 View Job Description"):
                st.write(analysis_results['job_description'])
            
            # Feedback
            st.subheader("💡 Suggestions")
            if analysis_results['match_score'] > 75:
                st.success("Strong match! Consider highlighting your most relevant skills in a summary section.")
            elif analysis_results['match_score'] > 50:
                st.warning("Moderate match. Try aligning your skills with the job requirements.")
            else:
                st.error("Low match. Consider acquiring the missing skills or applying for a different role.")
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()

