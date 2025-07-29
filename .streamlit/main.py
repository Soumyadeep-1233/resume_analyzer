import streamlit as st
import fitz  # PyMuPDF
from docx import Document
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
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

# Initialize spaCy with fallback
nlp = None
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
except (OSError, ImportError):
    st.warning("⚠️ Advanced NLP features unavailable. Using basic text processing.")
    nlp = None

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

# CGPA Eligibility Rules
def get_eligible_jobs(cgpa):
    """Return eligible job roles based on CGPA"""
    if cgpa < 7.5:
        return []
    elif 7.5 <= cgpa < 8.0:
        return ["Project Manager"]
    elif 8.0 <= cgpa < 8.5:
        return ["Project Manager", "Data Scientist"]
    else:  # cgpa >= 8.5
        return ["Project Manager", "Data Scientist", "Software Engineer"]

def get_cgpa_status_color(cgpa):
    """Return color based on CGPA range"""
    if cgpa < 7.5:
        return "error"
    elif cgpa < 8.0:
        return "warning"
    elif cgpa < 8.5:
        return "info"
    else:
        return "success"

# Preprocessing functions
def clean_text(text):
    """Normalize text"""
    text = unidecode(text)  # Convert accented characters
    text = re.sub(r'\W', ' ', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text.lower()

def extract_skills(text):
    """Extract technical skills using spaCy or fallback method"""
    skills = set()
    
    # Try spaCy first if available
    if nlp is not None:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["SKILL", "ORG", "PRODUCT"]:  # Broader entity types
                    skills.add(ent.text.lower())
        except Exception:
            pass  # Fall back to manual extraction
    
    # Enhanced manual skill extraction
    skill_keywords = [
        # Programming Languages
        "python", "java", "javascript", "c++", "c#", "r", "php", "swift", "kotlin",
        "typescript", "go", "rust", "scala", "ruby", "perl",
        
        # Databases
        "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "oracle",
        "sqlite", "cassandra", "dynamodb",
        
        # Machine Learning & AI
        "machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn",
        "keras", "opencv", "nlp", "computer vision", "neural networks", "ai",
        
        # Cloud & DevOps
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform",
        "ansible", "chef", "puppet", "ci/cd", "devops",
        
        # Web Technologies
        "react", "angular", "vue", "node.js", "express", "django", "flask",
        "spring", "html", "css", "bootstrap", "jquery",
        
        # Data & Analytics
        "tableau", "power bi", "excel", "pandas", "numpy", "matplotlib", "seaborn",
        "spark", "hadoop", "etl", "data analysis", "statistics",
        
        # Project Management
        "agile", "scrum", "kanban", "jira", "confluence", "project management",
        "waterfall", "lean", "six sigma",
        
        # Other Technologies
        "git", "github", "gitlab", "linux", "windows", "macos", "api", "rest",
        "graphql", "microservices", "blockchain", "iot"
    ]
    
    text_lower = text.lower()
    for skill in skill_keywords:
        if skill in text_lower:
            skills.add(skill)
    
    # Also look for patterns like "X years of experience in Y"
    experience_pattern = r'(\d+)\s+years?\s+(?:of\s+)?(?:experience\s+)?(?:in\s+|with\s+)?([a-zA-Z0-9\s\-\+#\.]+)'
    matches = re.findall(experience_pattern, text_lower)
    for years, skill in matches:
        skill = skill.strip()
        if len(skill) > 2 and len(skill) < 30:  # Reasonable skill name length
            skills.add(skill)
    
    return list(skills)

def extract_text_from_file(uploaded_file):
    """Extract text from PDF/DOCX"""
    text = ""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type == "pdf":
            # Reset file pointer
            uploaded_file.seek(0)
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                text = " ".join([page.get_text() for page in doc])
        elif file_type == "docx":
            # Reset file pointer
            uploaded_file.seek(0)
            doc = Document(BytesIO(uploaded_file.read()))
            text = " ".join([para.text for para in doc.paragraphs])
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
        if not text.strip():
            raise ValueError("No text could be extracted from the file")
            
    except Exception as e:
        st.error(f"Error extracting text from file: {str(e)}")
        return ""
    
    return clean_text(text)

def calculate_match(resume_text, job_description):
    """Calculate cosine similarity between resume and JD"""
    try:
        # Use TF-IDF instead of simple count for better results
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        vectors = vectorizer.fit_transform([resume_text, job_description])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return similarity * 100
    except Exception:
        # Fallback to basic method
        vectorizer = CountVectorizer()
        vectors = vectorizer.fit_transform([resume_text, job_description])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return similarity * 100

def analyze_resume(resume_text, selected_job):
    """Core analysis function"""
    if not resume_text.strip():
        return {
            "match_score": 0,
            "skills": [],
            "missing_skills": [],
            "job_description": JOB_DESCRIPTIONS[selected_job],
            "error": "No text found in resume"
        }
    
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
    st.title("📄 Resume Analyzer with CGPA Eligibility")
    st.write("Enter your CGPA and upload your resume to see job eligibility and match scores!")
    
    # Info about spaCy status
    if nlp is None:
        st.info("ℹ️ Running in basic mode. For enhanced NLP features, ensure spaCy and en_core_web_sm are properly installed.")
    
    # CGPA Input Section
    st.header("🎓 Academic Eligibility Check")
    
    cgpa_input = st.number_input(
        "Enter your CGPA (out of 10):",
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        value=0.0,
        help="Enter your Cumulative Grade Point Average"
    )
    
    if cgpa_input > 0:
        eligible_jobs = get_eligible_jobs(cgpa_input)
        
        # Display CGPA status
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Your CGPA", f"{cgpa_input}/10")
        
        with col2:
            st.metric("Eligible Positions", len(eligible_jobs))
        
        # Show eligibility status
        if cgpa_input < 7.5:
            st.error("❌ **Not Eligible**: Your CGPA is below the minimum requirement (7.5) for available positions.")
            st.info("💡 **Suggestion**: Consider improving your academic performance or exploring internship opportunities to gain experience.")
            return
        else:
            status_color = get_cgpa_status_color(cgpa_input)
            if status_color == "success":
                st.success(f"🌟 **Excellent!** You're eligible for all {len(eligible_jobs)} positions!")
            elif status_color == "info":
                st.info(f"👍 **Good!** You're eligible for {len(eligible_jobs)} positions.")
            else:
                st.warning(f"⚠️ **Fair**: You're eligible for {len(eligible_jobs)} position(s).")
        
        # Display eligible jobs
        st.subheader("✅ Eligible Job Roles:")
        for i, job in enumerate(eligible_jobs, 1):
            st.write(f"{i}. **{job}**")
        
        st.divider()
        
        # Resume Upload Section (only if eligible)
        st.header("📋 Resume Analysis")
        
        with st.sidebar:
            st.header("Settings")
            if eligible_jobs:
                selected_job = st.selectbox("Select Job Role:", eligible_jobs)
            else:
                selected_job = None
                st.error("No eligible positions based on CGPA")
            
            uploaded_file = st.file_uploader(
                "Upload Resume (PDF or DOCX):", 
                type=["pdf", "docx"],
                help="Upload your resume in PDF or DOCX format",
                disabled=(len(eligible_jobs) == 0)
            )
            
            if uploaded_file:
                st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        if uploaded_file and selected_job:
            with st.spinner("Analyzing your resume..."):
                try:
                    # Process resume
                    resume_text = extract_text_from_file(uploaded_file)
                    
                    if not resume_text:
                        st.error("❌ Could not extract text from the uploaded file. Please ensure it's a valid PDF or DOCX file with readable text.")
                        return
                    
                    analysis_results = analyze_resume(resume_text, selected_job)
                    
                    if "error" in analysis_results:
                        st.error(f"❌ {analysis_results['error']}")
                        return
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Match Score", f"{analysis_results['match_score']}%")
                    
                    with col2:
                        st.metric("Skills Found", len(analysis_results['skills']))
                    
                    with col3:
                        st.metric("Missing Skills", len(analysis_results['missing_skills']))
                    
                    st.divider()
                    
                    # Skills section
                    st.subheader("🔍 Skill Analysis")
                    
                    if analysis_results['skills']:
                        st.write("**✅ Your Skills:**")
                        # Display skills in a nice format
                        skills_text = ", ".join(analysis_results['skills'])
                        st.success(skills_text)
                    else:
                        st.write("**Your Skills:**")
                        st.warning("No specific skills detected. Try including more technical keywords in your resume.")
                    
                    if analysis_results['missing_skills']:
                        st.write("**❌ Skills to Improve:**")
                        missing_skills_text = ", ".join(analysis_results['missing_skills'])
                        st.error(missing_skills_text)
                    else:
                        st.write("**🎉 Great! You have all the key skills mentioned in the job description.**")
                    
                    st.divider()
                    
                    # Job Description preview
                    with st.expander("📝 View Job Description"):
                        st.write(analysis_results['job_description'])
                    
                    # Combined Feedback (CGPA + Skills)
                    st.subheader("💡 Overall Recommendations")
                    score = analysis_results['match_score']
                    
                    # CGPA-based feedback
                    st.write("**Academic Eligibility:**")
                    if cgpa_input >= 8.5:
                        st.success("🌟 **Excellent CGPA!** You meet the highest academic standards.")
                    elif cgpa_input >= 8.0:
                        st.info("👍 **Good CGPA!** You have strong academic credentials.")
                    else:
                        st.warning("⚠️ **Fair CGPA**: Consider highlighting other strengths in your application.")
                    
                    # Skills-based feedback
                    st.write("**Resume-Job Match:**")
                    if score >= 80:
                        st.success("🌟 **Excellent Match!** Your resume aligns very well with this role. Consider highlighting your most relevant achievements in a summary section.")
                    elif score >= 60:
                        st.warning("⚠️ **Good Match** - Consider emphasizing the skills that match the job requirements and adding any missing key skills to improve your chances.")
                    elif score >= 40:
                        st.warning("📈 **Moderate Match** - Try to better align your resume with the job requirements. Focus on the missing skills and consider gaining experience in those areas.")
                    else:
                        st.error("❌ **Low Match** - This role might not be the best fit based on your current resume. Consider acquiring the missing skills or applying for roles that better match your background.")
                    
                    # Combined recommendation
                    st.write("**Overall Assessment:**")
                    if cgpa_input >= 8.5 and score >= 70:
                        st.success("🚀 **Strong Candidate!** You have both excellent academics and relevant skills. Focus on crafting a compelling cover letter.")
                    elif cgpa_input >= 8.0 and score >= 60:
                        st.info("👌 **Good Candidate**: You're well-positioned for this role. Consider networking and showcasing your projects.")
                    else:
                        st.warning("📚 **Room for Improvement**: Focus on skill development and gaining relevant experience through projects or internships.")
                    
                    # Additional tips
                    with st.expander("📚 Tips to Improve Your Profile"):
                        st.markdown(f"""
                        **Academic Tips (Current CGPA: {cgpa_input}):**
                        {"- Maintain your excellent academic performance!" if cgpa_input >= 8.5 else ""}
                        {"- Consider taking advanced courses in your field" if cgpa_input < 8.5 else ""}
                        {"- Focus on improving grades in core subjects" if cgpa_input < 8.0 else ""}
                        
                        **Resume Tips:**
                        - Use keywords from the job description throughout your resume
                        - Quantify your achievements with numbers and metrics
                        - Tailor your resume for each specific job application
                        - Include a professional summary highlighting your key skills
                        - List your technical skills in a dedicated section
                        
                        **For Better Matching:**
                        - Mirror the language used in the job posting
                        - Include relevant certifications and courses
                        - Highlight projects that demonstrate required skills
                        - Use action verbs to describe your accomplishments
                        
                        **Career Development:**
                        - Build a portfolio of relevant projects
                        - Contribute to open-source projects (for tech roles)
                        - Gain certifications in missing skill areas
                        - Network with professionals in your target industry
                        """)
                    
                except Exception as e:
                    st.error(f"❌ An error occurred while processing your resume: {str(e)}")
                    st.info("Please try uploading a different file or contact support if the issue persists.")
    
    else:
        st.info("👆 Please enter your CGPA to check job eligibility.")

# Deployment Instructions
def show_deployment_info():
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Deployment Guide")
    with st.sidebar.expander("How to Deploy"):
        st.markdown("""
        **Prerequisites:**
        ```bash
        pip install streamlit
        pip install PyMuPDF
        pip install python-docx
        pip install pandas
        pip install nltk
        pip install scikit-learn
        pip install unidecode
        pip install spacy
        python -m spacy download en_core_web_sm
        ```
        
        **Run locally:**
        ```bash
        streamlit run app.py
        ```
        
        **Deploy to Streamlit Cloud:**
        1. Push code to GitHub
        2. Connect to streamlit.io
        3. Add requirements.txt with dependencies
        4. Deploy!
        """)

if __name__ == "__main__":
    main()
    show_deployment_info()

