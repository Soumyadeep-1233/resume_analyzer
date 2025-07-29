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
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Resume Analyzer | Job Match Score",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for storing candidate data
if 'candidates' not in st.session_state:
    st.session_state.candidates = []

# Initialize spaCy with fallback
nlp = None
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
except (OSError, ImportError):
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

def calculate_overall_score(cgpa, match_score, job_role):
    """Calculate overall candidate score with CGPA as priority"""
    # CGPA weight: 60%, Skills match weight: 40%
    cgpa_weight = 0.6
    skills_weight = 0.4
    
    # Normalize CGPA to 100 scale
    cgpa_score = (cgpa / 10) * 100
    
    # Apply job-specific CGPA bonus
    if job_role == "Software Engineer" and cgpa >= 8.5:
        cgpa_score += 5  # Bonus for top performers in SE
    elif job_role == "Data Scientist" and cgpa >= 8.0:
        cgpa_score += 3  # Bonus for DS
    
    overall_score = (cgpa_score * cgpa_weight) + (match_score * skills_weight)
    return round(overall_score, 2)

def rank_candidates_by_job(candidates, job_role):
    """Rank candidates for a specific job role"""
    # Filter eligible candidates for the job role
    eligible_candidates = []
    
    for candidate in candidates:
        eligible_jobs = get_eligible_jobs(candidate['cgpa'])
        if job_role in eligible_jobs:
            # Calculate overall score for this job role
            overall_score = calculate_overall_score(
                candidate['cgpa'], 
                candidate['analysis_results'].get('match_scores', {}).get(job_role, 0),
                job_role
            )
            
            eligible_candidates.append({
                'username': candidate['username'],
                'cgpa': candidate['cgpa'],
                'match_score': candidate['analysis_results'].get('match_scores', {}).get(job_role, 0),
                'overall_score': overall_score,
                'skills': candidate['analysis_results'].get('skills', []),
                'missing_skills': candidate['analysis_results'].get('missing_skills_by_job', {}).get(job_role, [])
            })
    
    # Sort by overall score (CGPA priority built into the score)
    eligible_candidates.sort(key=lambda x: x['overall_score'], reverse=True)
    
    return eligible_candidates

def analyze_resume_for_all_jobs(resume_text, cgpa):
    """Analyze resume against all job roles"""
    eligible_jobs = get_eligible_jobs(cgpa)
    
    if not eligible_jobs:
        return {
            "match_scores": {},
            "skills": [],
            "missing_skills_by_job": {},
            "eligible_jobs": []
        }
    
    # Extract skills from resume
    skills = extract_skills(resume_text)
    match_scores = {}
    missing_skills_by_job = {}
    
    # Analyze for each eligible job
    for job_role in eligible_jobs:
        jd_text = clean_text(JOB_DESCRIPTIONS[job_role])
        match_score = round(calculate_match(resume_text, jd_text), 1)
        match_scores[job_role] = match_score
        
        # Get missing skills for this job
        jd_skills = extract_skills(jd_text)
        missing_skills_by_job[job_role] = [skill for skill in jd_skills if skill not in skills]
    
    return {
        "match_scores": match_scores,
        "skills": skills,
        "missing_skills_by_job": missing_skills_by_job,
        "eligible_jobs": eligible_jobs
    }

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

# Streamlit UI
def main():
    st.title("📄 Resume Analyzer with Multi-Candidate Ranking")
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["👤 Individual Analysis", "🏆 Candidate Rankings"])
    
    with tab1:
        st.header("Individual Resume Analysis")
        individual_analysis()
    
    with tab2:
        st.header("Multi-Candidate Ranking System")
        ranking_system()

def individual_analysis():
    """Individual resume analysis interface"""
    st.write("Enter your details and upload your resume to see job eligibility and match scores!")
    
    # User details input
    col1, col2 = st.columns(2)
    
    with col1:
        username = st.text_input("Enter Username:", placeholder="e.g., john_doe")
    
    with col2:
        cgpa_input = st.number_input(
            "Enter CGPA (out of 10):",
            min_value=0.0,
            max_value=10.0,
            step=0.1,
            value=0.0,
            help="Enter your Cumulative Grade Point Average"
        )
    
    if username and cgpa_input > 0:
        eligible_jobs = get_eligible_jobs(cgpa_input)
        
        # Display CGPA status
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CGPA", f"{cgpa_input}/10")
        
        with col2:
            st.metric("Eligible Positions", len(eligible_jobs))
        
        # Show eligibility status
        if cgpa_input < 7.5:
            st.error("❌ **Not Eligible**: CGPA is below the minimum requirement (7.5) for available positions.")
            st.info("💡 **Suggestion**: Consider improving academic performance or exploring internship opportunities.")
            return
        else:
            status_color = get_cgpa_status_color(cgpa_input)
            if status_color == "success":
                st.success(f"🌟 **Excellent!** Eligible for all {len(eligible_jobs)} positions!")
            elif status_color == "info":
                st.info(f"👍 **Good!** Eligible for {len(eligible_jobs)} positions.")
            else:
                st.warning(f"⚠️ **Fair**: Eligible for {len(eligible_jobs)} position(s).")
        
        # Display eligible jobs
        st.subheader("✅ Eligible Job Roles:")
        for i, job in enumerate(eligible_jobs, 1):
            st.write(f"{i}. **{job}**")
        
        st.divider()
        
        # Resume Upload Section
        uploaded_file = st.file_uploader(
            "Upload Resume (PDF or DOCX):", 
            type=["pdf", "docx"],
            help="Upload your resume in PDF or DOCX format"
        )
        
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            with st.spinner("Analyzing your resume..."):
                try:
                    # Process resume
                    resume_text = extract_text_from_file(uploaded_file)
                    
                    if not resume_text:
                        st.error("❌ Could not extract text from the uploaded file.")
                        return
                    
                    # Analyze for all eligible jobs
                    analysis_results = analyze_resume_for_all_jobs(resume_text, cgpa_input)
                    
                    # Store candidate data
                    candidate_data = {
                        'username': username,
                        'cgpa': cgpa_input,
                        'analysis_results': analysis_results,
                        'timestamp': datetime.now()
                    }
                    
                    # Check if user already exists and update, otherwise add new
                    existing_index = None
                    for i, candidate in enumerate(st.session_state.candidates):
                        if candidate['username'] == username:
                            existing_index = i
                            break
                    
                    if existing_index is not None:
                        st.session_state.candidates[existing_index] = candidate_data
                        st.info(f"Updated data for user: {username}")
                    else:
                        st.session_state.candidates.append(candidate_data)
                        st.success(f"Added new candidate: {username}")
                    
                    # Display individual results
                    st.success("✅ Analysis Complete!")
                    
                    # Show results for each eligible job
                    for job_role in analysis_results['eligible_jobs']:
                        with st.expander(f"📊 {job_role} Analysis"):
                            match_score = analysis_results['match_scores'][job_role]
                            overall_score = calculate_overall_score(cgpa_input, match_score, job_role)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Match Score", f"{match_score}%")
                            with col2:
                                st.metric("Overall Score", f"{overall_score}%")
                            with col3:
                                st.metric("Missing Skills", len(analysis_results['missing_skills_by_job'][job_role]))
                            
                            if analysis_results['missing_skills_by_job'][job_role]:
                                st.write("**Skills to Improve:**")
                                st.error(", ".join(analysis_results['missing_skills_by_job'][job_role]))
                            else:
                                st.write("**🎉 Great! You have all the key skills for this role.**")
                    
                    # Show detected skills
                    if analysis_results['skills']:
                        st.subheader("🔍 Your Skills")
                        st.success(", ".join(analysis_results['skills']))
                    
                except Exception as e:
                    st.error(f"❌ An error occurred while processing your resume: {str(e)}")

def ranking_system():
    """Multi-candidate ranking interface"""
    if not st.session_state.candidates:
        st.info("📝 No candidates analyzed yet. Please use the Individual Analysis tab to add candidates first.")
        return
    
    st.write(f"**Total Candidates Analyzed:** {len(st.session_state.candidates)}")
    
    # Job selection for ranking
    selected_job_for_ranking = st.selectbox(
        "Select Job Role for Ranking:",
        ["Project Manager", "Data Scientist", "Software Engineer"],
        help="View candidate rankings for the selected job role"
    )
    
    if selected_job_for_ranking:
        # Get ranked candidates for the selected job
        ranked_candidates = rank_candidates_by_job(st.session_state.candidates, selected_job_for_ranking)
        
        if not ranked_candidates:
            st.warning(f"❌ No candidates are eligible for {selected_job_for_ranking} position.")
            st.info("Candidates need minimum CGPA requirements to be eligible for different positions.")
            return
        
        st.subheader(f"🏆 Candidate Rankings for {selected_job_for_ranking}")
        st.write(f"**Eligible Candidates:** {len(ranked_candidates)}")
        
        # Create ranking table
        ranking_data = []
        for rank, candidate in enumerate(ranked_candidates, 1):
            ranking_data.append({
                'Rank': rank,
                'Username': candidate['username'],
                'CGPA': candidate['cgpa'],
                'Match Score (%)': candidate['match_score'],
                'Overall Score (%)': candidate['overall_score'],
                'Skills Count': len(candidate['skills']),
                'Missing Skills': len(candidate['missing_skills'])
            })
        
        # Display ranking table
        df = pd.DataFrame(ranking_data)
        
        # Style the dataframe
        def highlight_top_candidates(row):
            if row['Rank'] == 1:
                return ['background-color: #d4edda; font-weight: bold'] * len(row)
            elif row['Rank'] == 2:
                return ['background-color: #f8f9fa; font-weight: bold'] * len(row)
            elif row['Rank'] == 3:
                return ['background-color: #f8f9fa'] * len(row)
            else:
                return [''] * len(row)
        
        styled_df = df.style.apply(highlight_top_candidates, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Detailed view for top candidates
        st.subheader("🔍 Top 3 Candidate Details")
        
        top_candidates = ranked_candidates[:3]
        for i, candidate in enumerate(top_candidates):
            with st.expander(f"🥇 Rank {i+1}: {candidate['username']} (Score: {candidate['overall_score']}%)"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("CGPA", f"{candidate['cgpa']}/10")
                with col2:
                    st.metric("Match Score", f"{candidate['match_score']}%")
                with col3:
                    st.metric("Overall Score", f"{candidate['overall_score']}%")
                with col4:
                    st.metric("Skills Found", len(candidate['skills']))
                
                if candidate['skills']:
                    st.write("**✅ Skills:**")
                    st.success(", ".join(candidate['skills'][:10]))  # Show first 10 skills
                
                if candidate['missing_skills']:
                    st.write("**❌ Missing Skills:**")
                    st.error(", ".join(candidate['missing_skills'][:5]))  # Show first 5 missing skills
        
        # Statistics
        st.subheader("📊 Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_cgpa = sum([c['cgpa'] for c in ranked_candidates]) / len(ranked_candidates)
            st.metric("Average CGPA", f"{avg_cgpa:.2f}")
        
        with col2:
            avg_match = sum([c['match_score'] for c in ranked_candidates]) / len(ranked_candidates)
            st.metric("Average Match Score", f"{avg_match:.1f}%")
        
        with col3:
            avg_overall = sum([c['overall_score'] for c in ranked_candidates]) / len(ranked_candidates)
            st.metric("Average Overall Score", f"{avg_overall:.1f}%")
    
    # Management options
    st.divider()
    st.subheader("🛠️ Management Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear All Candidates", type="secondary"):
            st.session_state.candidates = []
            st.success("All candidate data cleared!")
            st.rerun()
    
    with col2:
        if st.button("📊 Export Rankings", type="secondary"):
            if ranked_candidates:
                # Create export data
                export_data = []
                for rank, candidate in enumerate(ranked_candidates, 1):
                    export_data.append({
                        'Rank': rank,
                        'Username': candidate['username'],
                        'CGPA': candidate['cgpa'],
                        'Job_Role': selected_job_for_ranking,
                        'Match_Score': candidate['match_score'],
                        'Overall_Score': candidate['overall_score'],
                        'Skills_Count': len(candidate['skills']),
                        'Missing_Skills_Count': len(candidate['missing_skills']),
                        'Skills': '; '.join(candidate['skills']),
                        'Missing_Skills': '; '.join(candidate['missing_skills'])
                    })
                
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"{selected_job_for_ranking}_rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No data to export!")
    
    # Show all candidates summary
    if st.session_state.candidates:
        st.divider()
        st.subheader("👥 All Candidates Summary")
        
        summary_data = []
        for candidate in st.session_state.candidates:
            eligible_jobs = get_eligible_jobs(candidate['cgpa'])
            summary_data.append({
                'Username': candidate['username'],
                'CGPA': candidate['cgpa'],
                'Eligible Jobs': len(eligible_jobs),
                'Job Roles': ', '.join(eligible_jobs) if eligible_jobs else 'None',
                'Added On': candidate['timestamp'].strftime('%Y-%m-%d %H:%M')
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

if __name__ == "__main__":
    main()

