"""
Advanced Recruitment Intelligence Platform
A comprehensive AI-powered recruitment system with advanced analytics,
talent mapping, and predictive hiring insights.
"""

import streamlit as st
st.set_page_config(
    page_title="Smart Recruitment Platform", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Next-Gen Recruitment Intelligence Platform"
    }
)

import io, os, re, sqlite3, json, hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import math
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----- Enhanced Import Management -----
class DependencyManager:
    """Manages optional dependencies with graceful fallbacks"""
    
    def __init__(self):
        self.dependencies = {}
        self.missing = []
        self._load_dependencies()
    
    def _load_dependencies(self):
        deps = {
            'pdfplumber': None,
            'docx2txt': None,
            'rapidfuzz': None,
            'sentence_transformers': None,
            'pandas': None,
            'plotly': None,
            'fpdf': None,
            'numpy': None,
            'sklearn': None
        }
        
        for dep_name in deps:
            try:
                if dep_name == 'sentence_transformers':
                    from sentence_transformers import SentenceTransformer, util
                    self.dependencies['SentenceTransformer'] = SentenceTransformer
                    self.dependencies['util'] = util
                elif dep_name == 'rapidfuzz':
                    from rapidfuzz import fuzz
                    self.dependencies['fuzz'] = fuzz
                elif dep_name == 'plotly':
                    import plotly.graph_objects as go
                    import plotly.express as px
                    self.dependencies['go'] = go
                    self.dependencies['px'] = px
                elif dep_name == 'sklearn':
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.metrics.pairwise import cosine_similarity
                    self.dependencies['TfidfVectorizer'] = TfidfVectorizer
                    self.dependencies['cosine_similarity'] = cosine_similarity
                else:
                    module = __import__(dep_name)
                    self.dependencies[dep_name] = module
            except ImportError:
                self.missing.append(dep_name)
                logger.warning(f"Optional dependency {dep_name} not available")
    
    def get(self, name):
        return self.dependencies.get(name)
    
    def check_missing(self):
        if self.missing:
            return True, self.missing
        return False, []

deps = DependencyManager()

# Check dependencies
has_missing, missing_deps = deps.check_missing()
if has_missing:
    st.sidebar.warning(f"⚠️ Optional features unavailable: {', '.join(missing_deps)}")
    st.sidebar.caption(f"Install: `pip install {' '.join(missing_deps)}`")

pd = deps.get('pandas')
pdfplumber = deps.get('pdfplumber')
docx2txt = deps.get('docx2txt')
fuzz = deps.get('fuzz')
SentenceTransformer = deps.get('SentenceTransformer')
util = deps.get('util')
go = deps.get('go')
px = deps.get('px')
FPDF = deps.get('fpdf')
np = deps.get('numpy')
TfidfVectorizer = deps.get('TfidfVectorizer')
cosine_similarity = deps.get('cosine_similarity')

# ----- Data Models -----
class MatchLevel(Enum):
    EXCELLENT = "Excellent Match"
    STRONG = "Strong Match"
    GOOD = "Good Match"
    MODERATE = "Moderate Match"
    WEAK = "Weak Match"
    
@dataclass
class JobDescription:
    id: int
    title: str
    department: str
    raw_text: str
    must_have_skills: List[str]
    good_to_have_skills: List[str]
    experience_range: Tuple[int, int]
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CandidateProfile:
    name: str
    email: str
    phone: str
    resume_text: str
    skills: List[str]
    experience_years: int
    education: List[str]
    certifications: List[str]
    
@dataclass
class EvaluationResult:
    candidate: CandidateProfile
    job: JobDescription
    technical_score: float
    cultural_fit_score: float
    experience_score: float
    semantic_score: float
    final_score: float
    match_level: MatchLevel
    strengths: List[str]
    gaps: List[str]
    recommendations: List[str]
    interview_questions: List[str]
    risk_factors: List[str]

# ----- Enhanced Database Layer -----
class DatabaseManager:
    """Enhanced database management with migration support"""
    
    def __init__(self, db_path: str = "talentiq.db"):
        self.db_path = db_path
        self.conn = None
        self.init_database()
    
    def init_database(self):
        """Initialize database with enhanced schema"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = self.conn.cursor()
        
        # Enhanced JD table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS job_descriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            department TEXT,
            company TEXT,
            location TEXT,
            employment_type TEXT,
            salary_range TEXT,
            raw_text TEXT,
            must_have_skills TEXT,
            good_to_have_skills TEXT,
            experience_min INTEGER,
            experience_max INTEGER,
            metadata TEXT,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        
        # Enhanced candidate evaluations
        cur.execute("""
        CREATE TABLE IF NOT EXISTS candidate_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER,
            candidate_name TEXT,
            candidate_email TEXT,
            candidate_phone TEXT,
            resume_hash TEXT,
            resume_text TEXT,
            technical_score REAL,
            cultural_fit_score REAL,
            experience_score REAL,
            semantic_score REAL,
            final_score REAL,
            match_level TEXT,
            strengths TEXT,
            gaps TEXT,
            recommendations TEXT,
            interview_questions TEXT,
            risk_factors TEXT,
            recruiter_notes TEXT,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES job_descriptions(id)
        )""")
        
        # Analytics table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS recruitment_analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER,
            metric_type TEXT,
            metric_value REAL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES job_descriptions(id)
        )""")
        
        # Interview feedback
        cur.execute("""
        CREATE TABLE IF NOT EXISTS interview_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluation_id INTEGER,
            interviewer TEXT,
            round TEXT,
            rating INTEGER,
            comments TEXT,
            recommendation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (evaluation_id) REFERENCES candidate_evaluations(id)
        )""")
        
        self.conn.commit()
    
    def __del__(self):
        if self.conn:
            self.conn.close()

db = DatabaseManager()

# ----- Advanced Text Processing -----
class TextProcessor:
    """Advanced text processing and extraction"""
    
    SKILL_PATTERNS = {
        'programming': r'\b(python|java|c\+\+|c#|javascript|typescript|go|rust|ruby|php|swift|kotlin|scala|r)\b',
        'frameworks': r'\b(react|angular|vue|django|flask|spring|express|fastapi|rails|laravel)\b',
        'databases': r'\b(sql|mysql|postgresql|mongodb|redis|cassandra|elasticsearch|dynamodb)\b',
        'cloud': r'\b(aws|azure|gcp|cloud|lambda|ec2|s3|kubernetes|docker|terraform)\b',
        'ai_ml': r'\b(machine learning|deep learning|ai|tensorflow|pytorch|scikit-learn|nlp|computer vision)\b',
        'devops': r'\b(ci/cd|jenkins|gitlab|github actions|ansible|puppet|chef|monitoring|logging)\b'
    }
    
    EDUCATION_PATTERNS = {
        'degree': r'\b(bachelor|master|phd|mba|b\.?tech|m\.?tech|b\.?e|m\.?e|bsc|msc)\b',
        'university': r'\b(university|college|institute|school)\b'
    }
    
    @staticmethod
    def extract_text_from_file(uploaded_file) -> str:
        """Enhanced file extraction with better error handling"""
        if not uploaded_file:
            return ""
        
        filename = uploaded_file.name.lower()
        try:
            data = uploaded_file.read()
        except:
            try:
                data = uploaded_file.getvalue()
            except:
                logger.error(f"Failed to read file: {filename}")
                return ""
        
        extractors = {
            '.pdf': TextProcessor._extract_pdf,
            '.docx': TextProcessor._extract_docx,
            '.txt': TextProcessor._extract_text
        }
        
        for ext, extractor in extractors.items():
            if filename.endswith(ext):
                return extractor(data)
        
        # Default to text extraction
        return TextProcessor._extract_text(data)
    
    @staticmethod
    def _extract_pdf(data: bytes) -> str:
        if not pdfplumber:
            return ""
        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                text = "\n".join([p.extract_text() or "" for p in pdf.pages])
            return text
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return ""
    
    @staticmethod
    def _extract_docx(data: bytes) -> str:
        if not docx2txt:
            return ""
        temp_file = f"temp_{datetime.now().timestamp()}.docx"
        try:
            with open(temp_file, "wb") as f:
                f.write(data)
            text = docx2txt.process(temp_file)
            return text or ""
        except Exception as e:
            logger.error(f"DOCX extraction error: {e}")
            return ""
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    @staticmethod
    def _extract_text(data: bytes) -> str:
        try:
            return data.decode('utf-8', errors='ignore')
        except:
            return ""
    
    @staticmethod
    def clean_and_normalize(text: str) -> str:
        """Advanced text cleaning and normalization"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\.\,\;\:\@\#\+]', ' ', text)
        
        # Normalize case for better matching
        lines = text.split('\n')
        normalized = []
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue
            
            # Skip repetitive headers/footers
            if line.lower() in ['page', 'resume', 'cv', 'curriculum vitae']:
                continue
            
            normalized.append(line)
        
        return ' '.join(normalized)
    
    @staticmethod
    def extract_contact_info(text: str) -> Dict[str, str]:
        """Extract contact information from resume"""
        contact = {
            'email': '',
            'phone': '',
            'linkedin': '',
            'github': ''
        }
        
        # Email pattern
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if email_match:
            contact['email'] = email_match.group()
        
        # Phone pattern (various formats)
        phone_match = re.search(r'(\+\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}', text)
        if phone_match:
            contact['phone'] = phone_match.group()
        
        # LinkedIn
        linkedin_match = re.search(r'linkedin\.com/in/[\w\-]+', text, re.IGNORECASE)
        if linkedin_match:
            contact['linkedin'] = linkedin_match.group()
        
        # GitHub
        github_match = re.search(r'github\.com/[\w\-]+', text, re.IGNORECASE)
        if github_match:
            contact['github'] = github_match.group()
        
        return contact
    
    @staticmethod
    def extract_experience_years(text: str) -> int:
        """Extract years of experience from resume"""
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience[:\s]+(\d+)\+?\s*years?',
            r'(\d+)\s*years?\s*(?:of\s*)?professional'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # Fallback: count years mentioned in employment history
        year_mentions = re.findall(r'\b(19|20)\d{2}\b', text)
        if len(year_mentions) >= 2:
            years = [int(y) for y in year_mentions]
            return max(years) - min(years)
        
        return 0
    
    @staticmethod
    def extract_skills(text: str) -> Dict[str, List[str]]:
        """Extract categorized skills from text"""
        skills = {}
        text_lower = text.lower()
        
        for category, pattern in TextProcessor.SKILL_PATTERNS.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                skills[category] = list(set(matches))
        
        return skills
    
    @staticmethod
    def extract_education(text: str) -> List[str]:
        """Extract education information"""
        education = []
        text_lower = text.lower()
        
        for match in re.finditer(TextProcessor.EDUCATION_PATTERNS['degree'], text_lower):
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end]
            education.append(context.strip())
        
        return education

# ----- Advanced Matching Engine -----
class MatchingEngine:
    """Advanced candidate-job matching with multiple algorithms"""
    
    def __init__(self):
        self.embedder = self._load_embedder()
        self.tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 3)) if TfidfVectorizer else None
    
    def _load_embedder(self):
        """Load sentence transformer model with caching"""
        if not SentenceTransformer:
            return None
        
        try:
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return None
    
    def calculate_technical_match(self, jd_skills: List[str], candidate_skills: List[str]) -> Tuple[float, List[str], List[str]]:
        """Calculate technical skills match with fuzzy matching"""
        if not jd_skills:
            return 0.0, [], []
        
        matched = []
        unmatched = []
        
        for jd_skill in jd_skills:
            found = False
            jd_lower = jd_skill.lower()
            
            # Exact match
            for c_skill in candidate_skills:
                if jd_lower == c_skill.lower():
                    matched.append(jd_skill)
                    found = True
                    break
            
            # Fuzzy match
            if not found and fuzz:
                for c_skill in candidate_skills:
                    if fuzz.ratio(jd_lower, c_skill.lower()) >= 85:
                        matched.append(jd_skill)
                        found = True
                        break
            
            if not found:
                unmatched.append(jd_skill)
        
        score = (len(matched) / len(jd_skills)) * 100 if jd_skills else 0
        return round(score, 2), matched, unmatched
    
    def calculate_semantic_similarity(self, jd_text: str, resume_text: str) -> float:
        """Calculate semantic similarity using embeddings or TF-IDF"""
        if not jd_text or not resume_text:
            return 0.0
        
        # Try embeddings first
        if self.embedder and util:
            try:
                jd_emb = self.embedder.encode(jd_text[:2000], convert_to_tensor=True)
                resume_emb = self.embedder.encode(resume_text[:2000], convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(jd_emb, resume_emb).item() * 100
                return round(similarity, 2)
            except:
                pass
        
        # Fallback to TF-IDF
        if self.tfidf and cosine_similarity:
            try:
                vectors = self.tfidf.fit_transform([jd_text[:2000], resume_text[:2000]])
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100
                return round(similarity, 2)
            except:
                pass
        
        # Basic keyword overlap
        jd_words = set(jd_text.lower().split())
        resume_words = set(resume_text.lower().split())
        overlap = len(jd_words & resume_words) / len(jd_words) if jd_words else 0
        return round(overlap * 100, 2)
    
    def calculate_experience_match(self, required_range: Tuple[int, int], candidate_years: int) -> float:
        """Calculate experience match score"""
        min_years, max_years = required_range
        
        if candidate_years < min_years:
            # Under-qualified
            score = max(0, 100 - (min_years - candidate_years) * 20)
        elif candidate_years > max_years:
            # Over-qualified (slight penalty)
            score = max(70, 100 - (candidate_years - max_years) * 5)
        else:
            # Perfect match
            score = 100
        
        return round(score, 2)
    
    def calculate_cultural_fit(self, resume_text: str) -> float:
        """Estimate cultural fit based on soft skills and values"""
        cultural_keywords = {
            'teamwork': ['team', 'collaboration', 'cooperat', 'together'],
            'leadership': ['lead', 'manage', 'mentor', 'guide', 'coach'],
            'innovation': ['innovat', 'creative', 'novel', 'pioneer', 'transform'],
            'communication': ['communicat', 'present', 'document', 'articulate'],
            'problem_solving': ['problem', 'solution', 'analytical', 'troubleshoot'],
            'adaptability': ['adapt', 'flexible', 'agile', 'versatile'],
            'growth': ['learn', 'grow', 'develop', 'improve', 'progress']
        }
        
        text_lower = resume_text.lower()
        score = 0
        max_score = len(cultural_keywords) * 10
        
        for category, keywords in cultural_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    score += 10
                    break
        
        return round((score / max_score) * 100, 2)
    
    def determine_match_level(self, score: float) -> MatchLevel:
        """Determine match level based on final score"""
        if score >= 85:
            return MatchLevel.EXCELLENT
        elif score >= 75:
            return MatchLevel.STRONG
        elif score >= 65:
            return MatchLevel.GOOD
        elif score >= 50:
            return MatchLevel.MODERATE
        else:
            return MatchLevel.WEAK
    
    def generate_risk_factors(self, candidate: CandidateProfile, job: JobDescription, gaps: List[str]) -> List[str]:
        """Identify potential risk factors"""
        risks = []
        
        # Critical skill gaps
        critical_gaps = [g for g in gaps if g in job.must_have_skills]
        if len(critical_gaps) > 3:
            risks.append(f"Missing {len(critical_gaps)} critical skills")
        
        # Experience mismatch
        if candidate.experience_years < job.experience_range[0]:
            deficit = job.experience_range[0] - candidate.experience_years
            risks.append(f"Experience deficit of {deficit} years")
        elif candidate.experience_years > job.experience_range[1] + 5:
            risks.append("Significantly overqualified - retention risk")
        
        # Job hopping
        if 'month' in candidate.resume_text.lower():
            month_counts = len(re.findall(r'\d+\s*months?', candidate.resume_text.lower()))
            if month_counts > 3:
                risks.append("Potential job hopping pattern detected")
        
        return risks

# ----- Interview Intelligence -----
class InterviewIntelligence:
    """Generate intelligent interview questions and assessment criteria"""
    
    QUESTION_TEMPLATES = {
        'technical': {
            'python': [
                "Explain the difference between deep and shallow copy in Python. Provide examples.",
                "How would you optimize a Python application that's running slowly?",
                "Describe decorators and their use cases with a practical example."
            ],
            'javascript': [
                "Explain event delegation and bubbling in JavaScript.",
                "What are closures and how do they work? Provide a use case.",
                "How would you handle asynchronous operations in modern JavaScript?"
            ],
            'system_design': [
                "Design a scalable URL shortening service like bit.ly",
                "How would you architect a real-time chat application?",
                "Explain how you'd design a distributed caching system."
            ],
            'database': [
                "Explain database indexing strategies and their trade-offs.",
                "How would you optimize a slow-running query?",
                "Describe ACID properties with real-world examples."
            ]
        },
        'behavioral': [
            "Describe a challenging technical problem you solved. What was your approach?",
            "Tell me about a time you had to learn a new technology quickly.",
            "How do you handle disagreements with team members about technical decisions?",
            "Describe a situation where you had to balance technical debt with feature delivery."
        ],
        'situational': [
            "Your team's deployment fails in production. How do you handle it?",
            "A stakeholder requests a feature that you know is technically unfeasible. How do you respond?",
            "You discover a security vulnerability in legacy code. What steps do you take?"
        ]
    }
    
    @classmethod
    def generate_questions(cls, skills_gap: List[str], matched_skills: List[str], 
                          seniority_level: str = "mid") -> List[str]:
        """Generate personalized interview questions"""
        questions = []
        
        # Technical questions based on matched skills
        for skill in matched_skills[:3]:
            skill_lower = skill.lower()
            for category, q_list in cls.QUESTION_TEMPLATES['technical'].items():
                if category in skill_lower or skill_lower in category:
                    questions.extend(q_list[:1])
                    break
        
        # Gap assessment questions
        for gap in skills_gap[:2]:
            questions.append(f"How would you approach learning {gap}? What resources would you use?")
        
        # Add behavioral questions
        questions.extend(cls.QUESTION_TEMPLATES['behavioral'][:2])
        
        # Add situational questions based on seniority
        if seniority_level in ["senior", "lead"]:
            questions.extend(cls.QUESTION_TEMPLATES['situational'][:2])
        
        return questions[:8]  # Limit to 8 questions

# ----- Analytics Engine -----
class AnalyticsEngine:
    """Advanced analytics and insights generation"""
    
    @staticmethod
    def generate_batch_insights(evaluations: pd.DataFrame) -> Dict[str, Any]:
        """Generate insights from batch evaluations"""
        if evaluations.empty:
            return {}
        
        insights = {
            'total_candidates': len(evaluations),
            'avg_match_score': evaluations['final_score'].mean(),
            'match_distribution': evaluations['match_level'].value_counts().to_dict(),
            'top_skills': {},
            'common_gaps': {},
            'recommendations': []
        }
        
        # Score distribution
        score_ranges = {
            'Excellent (85-100)': len(evaluations[evaluations['final_score'] >= 85]),
            'Strong (75-84)': len(evaluations[(evaluations['final_score'] >= 75) & (evaluations['final_score'] < 85)]),
            'Good (65-74)': len(evaluations[(evaluations['final_score'] >= 65) & (evaluations['final_score'] < 75)]),
            'Moderate (50-64)': len(evaluations[(evaluations['final_score'] >= 50) & (evaluations['final_score'] < 65)]),
            'Weak (<50)': len(evaluations[evaluations['final_score'] < 50])
        }
        insights['score_distribution'] = score_ranges
        
        # Generate recommendations
        if insights['avg_match_score'] < 60:
            insights['recommendations'].append("Consider expanding sourcing channels or adjusting requirements")
        if score_ranges['Excellent (85-100)'] == 0:
            insights['recommendations'].append("No excellent matches found - review job requirements")
        
        return insights

# ----- Visualization Components -----
class Visualizer:
    """Enhanced visualization components"""
    
    @staticmethod
    def create_candidate_radar(scores: Dict[str, float], name: str):
        """Create radar chart for candidate profile"""
        if not go:
            st.info("Install plotly for visualizations")
            return None
        
        categories = list(scores.keys())
        values = list(scores.values())
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=name,
            line=dict(color='rgb(37, 99, 235)', width=2),
            fillcolor='rgba(37, 99, 235, 0.2)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=12)
                )
            ),
            showlegend=False,
            height=400,
            margin=dict(l=80, r=80, t=40, b=40)
        )
        
        return fig
    
    @staticmethod
    def create_skill_heatmap(candidates: List[str], skills: List[str], scores: Dict[str, Dict[str, float]]):
        """Create skill heatmap for batch comparison"""
        if not go or not pd:
            return None
        
        # Prepare data
        data = []
        for candidate in candidates:
            row = []
            for skill in skills:
                score = scores.get(candidate, {}).get(skill, 0)
                row.append(score)
            data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=skills,
            y=candidates,
            colorscale='Blues',
            text=data,
            texttemplate='%{text:.0f}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Skills Competency Heatmap",
            height=400 + len(candidates) * 30,
            xaxis_title="Skills",
            yaxis_title="Candidates"
        )
        
        return fig
    
    @staticmethod
    def create_pipeline_funnel(stage_counts: Dict[str, int]):
        """Create recruitment pipeline funnel"""
        if not go:
            return None
        
        stages = list(stage_counts.keys())
        counts = list(stage_counts.values())
        
        fig = go.Figure(go.Funnel(
            y=stages,
            x=counts,
            textposition="inside",
            textinfo="value+percent initial",
            opacity=0.8,
            marker=dict(
                color=["rgb(37, 99, 235)", "rgb(59, 130, 246)", 
                       "rgb(96, 165, 250)", "rgb(147, 197, 253)"]
            )
        ))
        
        fig.update_layout(
            title="Recruitment Pipeline",
            height=400,
            margin=dict(l=100, r=50, t=60, b=50)
        )
        
        return fig

# ----- Main Application UI -----
class TalentIQApp:
    """Main application controller"""
    
    def __init__(self):
        self.init_session_state()
        self.processor = TextProcessor()
        self.engine = MatchingEngine()
        self.interviewer = InterviewIntelligence()
        self.analytics = AnalyticsEngine()
        self.visualizer = Visualizer()
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'current_job_id' not in st.session_state:
            st.session_state.current_job_id = None
        if 'evaluation_results' not in st.session_state:
            st.session_state.evaluation_results = []
        if 'theme' not in st.session_state:
            st.session_state.theme = 'light'
    
    def render_header(self):
        """Render application header with branding"""
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown("""
                <div style="text-align: center; padding: 20px;">
                    <h1 style="color: #2563eb; font-size: 2.5rem; font-weight: bold;">
                           Aura Recruitment
                    </h1>
                    <p style="color: #64748b; font-size: 1.1rem;">
                        Advanced Recruitment Intelligence Platform
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if st.button("🌙" if st.session_state.theme == 'light' else "☀️"):
                st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
                st.rerun()
    
    def render_sidebar(self):
        """Render sidebar with navigation and stats"""
        with st.sidebar:
            st.markdown("### 📊 Quick Stats")
            
            # Get stats from database
            cur = db.conn.cursor()
            total_jobs = cur.execute("SELECT COUNT(*) FROM job_descriptions WHERE status='active'").fetchone()[0]
            total_candidates = cur.execute("SELECT COUNT(*) FROM candidate_evaluations").fetchone()[0]
            avg_match = cur.execute("SELECT AVG(final_score) FROM candidate_evaluations").fetchone()[0] or 0
            
            col1, col2 = st.columns(2)
            col1.metric("Active Jobs", total_jobs)
            col2.metric("Candidates", total_candidates)
            st.metric("Avg Match Score", f"{avg_match:.1f}%")
            
            st.markdown("---")
            st.markdown("### 🔧 Configuration")
            
            # Scoring weights
            st.markdown("#### Scoring Weights")
            technical_weight = st.slider("Technical Skills", 0.0, 1.0, 0.4, 0.05, key="tech_weight")
            semantic_weight = st.slider("Semantic Match", 0.0, 1.0, 0.3, 0.05, key="sem_weight")
            experience_weight = st.slider("Experience", 0.0, 1.0, 0.2, 0.05, key="exp_weight")
            cultural_weight = st.slider("Cultural Fit", 0.0, 1.0, 0.1, 0.05, key="cult_weight")
            
            # Normalize weights
            total_weight = technical_weight + semantic_weight + experience_weight + cultural_weight
            if total_weight > 0:
                st.session_state.weights = {
                    'technical': technical_weight / total_weight,
                    'semantic': semantic_weight / total_weight,
                    'experience': experience_weight / total_weight,
                    'cultural': cultural_weight / total_weight
                }
            
            st.markdown("---")
            st.markdown("### 💡 Tips")
            st.info("""
                • Upload multiple resumes for batch processing
                • Use semantic matching for better results
                • Export reports for offline review
            """)
    
    def job_management_tab(self):
        """Job description management interface"""
        st.markdown("## 📋 Job Management")
        
        tab1, tab2, tab3 = st.tabs(["➕ Create New Job", "📂 Manage Existing", "📊 Job Analytics"])
        
        with tab1:
            self.create_job_form()
        
        with tab2:
            self.manage_jobs()
        
        with tab3:
            self.job_analytics()
    
    def create_job_form(self):
        """Create new job posting form"""
        st.markdown("### Create New Job Posting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input("Job Title *", placeholder="e.g., Senior Software Engineer")
            department = st.text_input("Department", placeholder="e.g., Engineering")
            company = st.text_input("Company", placeholder="e.g., TechCorp Inc.")
            location = st.text_input("Location", placeholder="e.g., San Francisco, CA")
        
        with col2:
            employment_type = st.selectbox("Employment Type", 
                ["Full-time", "Part-time", "Contract", "Internship"])
            salary_range = st.text_input("Salary Range", placeholder="e.g., $120k-$180k")
            exp_min = st.number_input("Min Experience (years)", min_value=0, max_value=20, value=2)
            exp_max = st.number_input("Max Experience (years)", min_value=0, max_value=30, value=5)
        
        # JD upload or paste
        st.markdown("### Job Description")
        upload_option = st.radio("Choose input method:", ["Upload File", "Paste Text"], horizontal=True)
        
        jd_text = ""
        if upload_option == "Upload File":
            jd_file = st.file_uploader("Upload JD", type=['pdf', 'docx', 'txt'])
            if jd_file:
                jd_text = self.processor.extract_text_from_file(jd_file)
        else:
            jd_text = st.text_area("Paste Job Description", height=200)
        
        # Skills extraction
        col1, col2 = st.columns(2)
        with col1:
            must_have = st.text_area("Must-Have Skills (comma-separated)", 
                placeholder="Python, Django, PostgreSQL, AWS")
        with col2:
            good_to_have = st.text_area("Good-to-Have Skills (comma-separated)", 
                placeholder="Docker, Kubernetes, React, CI/CD")
        
        # Auto-extract skills button
        if jd_text and st.button("🤖 Auto-Extract Skills from JD"):
            extracted_skills = self.processor.extract_skills(jd_text)
            all_skills = []
            for category, skills in extracted_skills.items():
                all_skills.extend(skills)
            
            # Suggest must-have and good-to-have
            if all_skills:
                mid = len(all_skills) // 2
                st.info(f"Found {len(all_skills)} skills. Suggested distribution:")
                st.write("**Must-Have:** " + ", ".join(all_skills[:mid]))
                st.write("**Good-to-Have:** " + ", ".join(all_skills[mid:]))
        
        # Save button
        if st.button("💾 Save Job Posting", type="primary"):
            if not title or not jd_text:
                st.error("Job title and description are required!")
            else:
                # Parse skills
                must_have_list = [s.strip() for s in must_have.split(",") if s.strip()]
                good_to_have_list = [s.strip() for s in good_to_have.split(",") if s.strip()]
                
                # Save to database
                cur = db.conn.cursor()
                metadata = json.dumps({
                    'created_by': 'admin',
                    'version': 1
                })
                
                cur.execute("""
                    INSERT INTO job_descriptions 
                    (title, department, company, location, employment_type, salary_range,
                     raw_text, must_have_skills, good_to_have_skills, experience_min, 
                     experience_max, metadata, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (title, department, company, location, employment_type, salary_range,
                      jd_text, json.dumps(must_have_list), json.dumps(good_to_have_list),
                      exp_min, exp_max, metadata, 'active'))
                
                db.conn.commit()
                job_id = cur.lastrowid
                st.session_state.current_job_id = job_id
                
                st.success(f"✅ Job posting created successfully! (ID: {job_id})")
                st.balloons()
    
    def manage_jobs(self):
        """Manage existing job postings"""
        st.markdown("### Existing Job Postings")
        
        # Load jobs from database
        query = """
            SELECT id, title, department, location, status, 
                   (SELECT COUNT(*) FROM candidate_evaluations WHERE job_id = jd.id) as candidates,
                   created_at
            FROM job_descriptions jd
            ORDER BY created_at DESC
        """
        
        if pd:
            df = pd.read_sql_query(query, db.conn)
            
            if not df.empty:
                # Add action column
                df['Actions'] = df['id'].apply(lambda x: f"Job_{x}")
                
                # Display with filters
                status_filter = st.selectbox("Filter by Status", ["All", "active", "archived"])
                if status_filter != "All":
                    df = df[df['status'] == status_filter]
                
                # Format dates
                df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d')
                
                # Display table
                st.dataframe(
                    df[['id', 'title', 'department', 'location', 'candidates', 'status', 'created_at']],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    job_id = st.selectbox("Select Job ID", df['id'].tolist())
                
                with col2:
                    if st.button("📝 Select for Evaluation"):
                        st.session_state.current_job_id = job_id
                        st.success(f"Selected Job ID: {job_id}")
                
                with col3:
                    if st.button("📊 View Analytics"):
                        st.session_state.view_job_analytics = job_id
            else:
                st.info("No job postings found. Create one to get started!")
        else:
            st.warning("Install pandas for better data management")
    
    def job_analytics(self):
        """Display job-specific analytics"""
        st.markdown("### Job Performance Analytics")
        
        if not st.session_state.current_job_id:
            st.info("Select a job first to view analytics")
            return
        
        job_id = st.session_state.current_job_id
        
        # Get job details
        cur = db.conn.cursor()
        job = cur.execute("SELECT * FROM job_descriptions WHERE id = ?", (job_id,)).fetchone()
        if not job:
            st.error("Job not found")
            return
        
        st.markdown(f"#### Analytics for: {job[1]}")
        
        # Get evaluation metrics
        metrics = cur.execute("""
            SELECT 
                COUNT(*) as total_candidates,
                AVG(final_score) as avg_score,
                MAX(final_score) as max_score,
                MIN(final_score) as min_score,
                COUNT(CASE WHEN final_score >= 75 THEN 1 END) as strong_matches
            FROM candidate_evaluations
            WHERE job_id = ?
        """, (job_id,)).fetchone()
        
        if metrics[0] > 0:
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Candidates", metrics[0])
            col2.metric("Avg Match Score", f"{metrics[1]:.1f}%")
            col3.metric("Best Match", f"{metrics[2]:.1f}%")
            col4.metric("Strong Matches", metrics[4])
            
            # Score distribution chart
            if pd and px:
                evals = pd.read_sql_query(
                    "SELECT final_score, candidate_name FROM candidate_evaluations WHERE job_id = ?",
                    db.conn, params=(job_id,)
                )
                
                if not evals.empty:
                    fig = px.histogram(
                        evals, x='final_score', nbins=20,
                        title="Score Distribution",
                        labels={'final_score': 'Match Score (%)', 'count': 'Number of Candidates'}
                    )
                    fig.update_traces(marker_color='rgb(37, 99, 235)')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Top candidates
                    st.markdown("#### 🏆 Top 5 Candidates")
                    top5 = evals.nlargest(5, 'final_score')
                    for idx, row in top5.iterrows():
                        st.write(f"**{row['candidate_name']}** - Match Score: {row['final_score']:.1f}%")
        else:
            st.info("No evaluations yet for this job posting")
    
    def evaluation_tab(self):
        """Candidate evaluation interface"""
        st.markdown("## 🎯 Candidate Evaluation")
        
        if not st.session_state.current_job_id:
            st.warning("⚠️ Please select a job posting first from the Job Management tab")
            return
        
        # Get current job details
        cur = db.conn.cursor()
        job = cur.execute("SELECT * FROM job_descriptions WHERE id = ?", 
                         (st.session_state.current_job_id,)).fetchone()
        
        st.info(f"📌 Evaluating for: **{job[1]}** | {job[3]} | {job[4]}")
        
        # Resume upload
        st.markdown("### Upload Resumes")
        resumes = st.file_uploader(
            "Upload candidate resumes (PDF, DOCX, TXT)",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="You can select multiple files for batch processing"
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                include_cultural = st.checkbox("Include Cultural Fit Analysis", value=True)
                generate_questions = st.checkbox("Generate Interview Questions", value=True)
            with col2:
                export_reports = st.checkbox("Generate PDF Reports", value=False)
                anonymous_mode = st.checkbox("Anonymous Evaluation Mode", value=False)
        
        # Evaluation button
        if st.button("🚀 Start Evaluation", type="primary", disabled=not resumes):
            self.process_evaluations(job, resumes, include_cultural, generate_questions, 
                                    export_reports, anonymous_mode)
    
    def process_evaluations(self, job, resumes, include_cultural, generate_questions, 
                           export_reports, anonymous_mode):
        """Process batch resume evaluations"""
        
        # Parse job requirements
        must_have = json.loads(job[8]) if job[8] else []
        good_to_have = json.loads(job[9]) if job[9] else []
        all_skills = must_have + good_to_have
        
        results = []
        progress = st.progress(0)
        status = st.empty()
        
        for idx, resume_file in enumerate(resumes):
            status.text(f"Processing {idx+1}/{len(resumes)}: {resume_file.name}")
            
            # Extract resume text
            resume_text = self.processor.extract_text_from_file(resume_file)
            cleaned_text = self.processor.clean_and_normalize(resume_text)
            
            # Extract candidate info
            contact = self.processor.extract_contact_info(cleaned_text)
            experience = self.processor.extract_experience_years(cleaned_text)
            skills = self.processor.extract_skills(cleaned_text)
            
            # Flatten skills
            candidate_skills = []
            for category, skill_list in skills.items():
                candidate_skills.extend(skill_list)
            
            # Calculate scores
            tech_score, matched, gaps = self.engine.calculate_technical_match(
                all_skills, candidate_skills
            )
            
            semantic_score = self.engine.calculate_semantic_similarity(
                job[7], cleaned_text
            )
            
            exp_score = self.engine.calculate_experience_match(
                (job[10], job[11]), experience
            )
            
            cultural_score = 0
            if include_cultural:
                cultural_score = self.engine.calculate_cultural_fit(cleaned_text)
            
            # Calculate weighted final score
            weights = st.session_state.get('weights', {
                'technical': 0.4,
                'semantic': 0.3,
                'experience': 0.2,
                'cultural': 0.1
            })
            
            final_score = (
                tech_score * weights['technical'] +
                semantic_score * weights['semantic'] +
                exp_score * weights['experience'] +
                cultural_score * weights['cultural']
            )
            
            match_level = self.engine.determine_match_level(final_score)
            
            # Generate recommendations
            recommendations = []
            if final_score >= 75:
                recommendations.append("Fast-track to technical interview")
            elif final_score >= 60:
                recommendations.append("Schedule screening call")
            else:
                recommendations.append("Review with hiring manager")
            
            # Generate interview questions
            questions = []
            if generate_questions:
                questions = self.interviewer.generate_questions(gaps, matched, "mid")
            
            # Risk factors
            risk_factors = self.engine.generate_risk_factors(
                CandidateProfile(
                    name=resume_file.name if anonymous_mode else contact.get('email', resume_file.name),
                    email=contact.get('email', ''),
                    phone=contact.get('phone', ''),
                    resume_text=cleaned_text,
                    skills=candidate_skills,
                    experience_years=experience,
                    education=[],
                    certifications=[]
                ),
                JobDescription(
                    id=job[0], title=job[1], department=job[2], raw_text=job[7],
                    must_have_skills=must_have, good_to_have_skills=good_to_have,
                    experience_range=(job[10], job[11]), created_at=datetime.now(),
                    metadata={}
                ),
                gaps
            )
            
            # Save to database
            candidate_name = f"Candidate_{idx+1}" if anonymous_mode else resume_file.name
            
            cur = db.conn.cursor()
            cur.execute("""
                INSERT INTO candidate_evaluations
                (job_id, candidate_name, candidate_email, candidate_phone, resume_hash,
                 resume_text, technical_score, cultural_fit_score, experience_score,
                 semantic_score, final_score, match_level, strengths, gaps,
                 recommendations, interview_questions, risk_factors, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job[0], candidate_name, contact.get('email', ''), contact.get('phone', ''),
                hashlib.md5(cleaned_text.encode()).hexdigest(), cleaned_text[:5000],
                tech_score, cultural_score, exp_score, semantic_score, final_score,
                match_level.value, json.dumps(matched[:10]), json.dumps(gaps[:10]),
                json.dumps(recommendations), json.dumps(questions),
                json.dumps(risk_factors), 'evaluated'
            ))
            db.conn.commit()
            
            results.append({
                'name': candidate_name,
                'scores': {
                    'Technical': tech_score,
                    'Semantic': semantic_score,
                    'Experience': exp_score,
                    'Cultural': cultural_score
                },
                'final_score': final_score,
                'match_level': match_level,
                'matched': matched,
                'gaps': gaps,
                'recommendations': recommendations,
                'questions': questions,
                'risks': risk_factors
            })
            
            progress.progress((idx + 1) / len(resumes))
        
        status.empty()
        progress.empty()
        
        # Display results
        self.display_evaluation_results(results, export_reports)
    
    def display_evaluation_results(self, results, export_reports):
        """Display evaluation results with visualizations"""
        
        st.success(f"✅ Successfully evaluated {len(results)} candidates!")
        
        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Summary metrics
        st.markdown("### 📊 Evaluation Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        avg_score = sum(r['final_score'] for r in results) / len(results)
        excellent = sum(1 for r in results if r['match_level'] == MatchLevel.EXCELLENT)
        strong = sum(1 for r in results if r['match_level'] == MatchLevel.STRONG)
        
        col1.metric("Average Match", f"{avg_score:.1f}%")
        col2.metric("Excellent Matches", excellent)
        col3.metric("Strong Matches", strong)
        col4.metric("Total Evaluated", len(results))
        
        # Top candidates
        st.markdown("### 🏆 Top Candidates")
        
        for idx, candidate in enumerate(results[:5], 1):
            with st.expander(f"{idx}. {candidate['name']} - {candidate['final_score']:.1f}% ({candidate['match_level'].value})"):
                
                # Scores radar chart
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    st.markdown("#### Match Scores")
                    for metric, score in candidate['scores'].items():
                        st.metric(metric, f"{score:.1f}%")
                
                with col2:
                    fig = self.visualizer.create_candidate_radar(
                        candidate['scores'],
                        candidate['name']
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                # Strengths and gaps
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### ✅ Strengths")
                    for skill in candidate['matched'][:5]:
                        st.write(f"• {skill}")
                
                with col2:
                    st.markdown("#### ⚠️ Gaps")
                    for gap in candidate['gaps'][:5]:
                        st.write(f"• {gap}")
                
                # Recommendations
                st.markdown("#### 💡 Recommendations")
                for rec in candidate['recommendations']:
                    st.info(rec)
                
                # Interview questions
                if candidate['questions']:
                    st.markdown("#### 🎤 Suggested Interview Questions")
                    for q in candidate['questions'][:5]:
                        st.write(f"• {q}")
                
                # Risk factors
                if candidate['risks']:
                    st.markdown("#### ⚠️ Risk Factors")
                    for risk in candidate['risks']:
                        st.warning(risk)
                
                # Export button
                if export_reports:
                    if st.button(f"📄 Export Report for {candidate['name']}", key=f"export_{idx}"):
                        st.info("Report export feature coming soon!")
        
        # Batch comparison
        if len(results) > 1:
            st.markdown("### 🔄 Candidate Comparison")
            
            # Prepare data for heatmap
            candidates = [r['name'] for r in results[:10]]
            skills = list(set(skill for r in results for skill in r['matched']))[:15]
            
            skill_scores = {}
            for r in results[:10]:
                skill_scores[r['name']] = {}
                for skill in skills:
                    skill_scores[r['name']][skill] = 100 if skill in r['matched'] else 0
            
            fig = self.visualizer.create_skill_heatmap(candidates, skills, skill_scores)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Export all results
        if pd and results:
            df_results = pd.DataFrame([{
                'Candidate': r['name'],
                'Final Score': r['final_score'],
                'Match Level': r['match_level'].value,
                'Technical': r['scores']['Technical'],
                'Semantic': r['scores']['Semantic'],
                'Experience': r['scores']['Experience'],
                'Cultural': r['scores']['Cultural']
            } for r in results])
            
            csv = df_results.to_csv(index=False)
            st.download_button(
                "📥 Download Results (CSV)",
                csv,
                "evaluation_results.csv",
                "text/csv",
                key='download-csv'
            )
    
    def analytics_dashboard_tab(self):
        """Analytics and insights dashboard"""
        st.markdown("## 📈 Analytics Dashboard")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Talent Pool", "Pipeline", "Insights"])
        
        with tab1:
            self.render_overview_analytics()
        
        with tab2:
            self.render_talent_pool_analytics()
        
        with tab3:
            self.render_pipeline_analytics()
        
        with tab4:
            self.render_insights()
    
    def render_overview_analytics(self):
        """Render overview analytics"""
        st.markdown("### 📊 Recruitment Overview")
        
        if not pd:
            st.warning("Install pandas for analytics features")
            return
        
        # Date range filter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Overview Start Date", datetime.now() - timedelta(days=30), key="overview_start_date")
        with col2:
            end_date = st.date_input("Overview End Date", datetime.now(), key="overview_end_date")

        # Normalize dates for SQLite query
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Load data
        query = """
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as evaluations,
                AVG(final_score) as avg_score
            FROM candidate_evaluations
            WHERE DATE(created_at) BETWEEN ? AND ?
            GROUP BY DATE(created_at)
            ORDER BY date
        """
        
        df = pd.read_sql_query(query, db.conn, params=(start_str, end_str))
        
        if not df.empty:
            # Time series chart
            if px:
                fig = px.line(df, x='date', y='evaluations',
                            title="Daily Evaluation Trend",
                            labels={'evaluations': 'Number of Evaluations', 'date': 'Date'})
                fig.update_traces(line_color='rgb(37, 99, 235)')
                st.plotly_chart(fig, use_container_width=True)
                
                # Score trend
                fig2 = px.line(df, x='date', y='avg_score',
                             title="Average Match Score Trend",
                             labels={'avg_score': 'Average Score (%)', 'date': 'Date'})
                fig2.update_traces(line_color='rgb(16, 185, 129)')
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data available for selected date range")
    
    def render_talent_pool_analytics(self):
        """Render talent pool analytics"""
        st.markdown("### 👥 Talent Pool Analysis")
        
        if not pd:
            return
        
        # Load candidate data
        query = """
            SELECT 
                match_level,
                COUNT(*) as count,
                AVG(technical_score) as avg_technical,
                AVG(semantic_score) as avg_semantic,
                AVG(experience_score) as avg_experience
            FROM candidate_evaluations
            GROUP BY match_level
        """
        
        df = pd.read_sql_query(query, db.conn)
        
        if not df.empty and px:
            # Match level distribution
            fig = px.pie(df, values='count', names='match_level',
                        title="Talent Distribution by Match Level")
            st.plotly_chart(fig, use_container_width=True)
            
            # Score comparison
            df_melted = df.melt(id_vars=['match_level'], 
                               value_vars=['avg_technical', 'avg_semantic', 'avg_experience'],
                               var_name='Score Type', value_name='Score')
            
            fig2 = px.bar(df_melted, x='match_level', y='Score', color='Score Type',
                         title="Average Scores by Match Level",
                         barmode='group')
            st.plotly_chart(fig2, use_container_width=True)
    
    def render_pipeline_analytics(self):
        """Render recruitment pipeline analytics"""
        st.markdown("### 🔄 Recruitment Pipeline")
        
        # Simulated pipeline data (you can replace with actual data)
        pipeline_data = {
            'Applied': 500,
            'Screened': 300,
            'Technical Interview': 100,
            'HR Interview': 50,
            'Offered': 20,
            'Accepted': 15
        }
        
        fig = self.visualizer.create_pipeline_funnel(pipeline_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Conversion metrics
        st.markdown("### Conversion Rates")
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Screen to Interview", "33.3%", "↑ 5%")
        col2.metric("Interview to Offer", "20%", "↓ 2%")
        col3.metric("Offer Acceptance", "75%", "→ 0%")
    
    def render_insights(self):
        """Render AI-generated insights"""
        st.markdown("### 🤖 AI-Powered Insights")
        
        if not pd:
            return
        
        # Load recent evaluation data
        df = pd.read_sql_query("""
            SELECT * FROM candidate_evaluations 
            ORDER BY created_at DESC 
             """, db.conn)
        
        if not df.empty:
            # Generate insights
            insights = self.analytics.generate_batch_insights(df)
            
            # Display key insights
            if insights:
                # Score distribution insight
                st.markdown("#### 📊 Score Distribution")
                for range_name, count in insights.get('score_distribution', {}).items():
                    if count > 0:
                        st.write(f"• {range_name}: **{count}** candidates")
                
                # Recommendations
                if insights.get('recommendations'):
                    st.markdown("#### 💡 Recommendations")
                    for rec in insights['recommendations']:
                        st.info(rec)
                
                # Talent gap analysis
                st.markdown("#### 🎯 Talent Gap Analysis")
                gaps_query = """
                    SELECT gaps, COUNT(*) as frequency
                    FROM candidate_evaluations
                    WHERE gaps IS NOT NULL AND gaps != '[]'
                    GROUP BY gaps
                    ORDER BY frequency DESC
                    LIMIT 5
                """
                gaps_df = pd.read_sql_query(gaps_query, db.conn)
                
                if not gaps_df.empty:
                    st.write("**Most Common Skill Gaps:**")
                    for _, row in gaps_df.iterrows():
                        try:
                            gap_list = json.loads(row['gaps'])
                            if gap_list:
                                st.write(f"• {', '.join(gap_list[:3])}")
                        except:
                            continue
                
                # Success factors
                st.markdown("#### ✨ Success Factors")
                top_performers = df[df['final_score'] >= 80]
                if not top_performers.empty:
                    st.write(f"**{len(top_performers)}** candidates scored above 80%")
                    st.write("Common attributes of top performers:")
                    st.write("• Strong technical skill alignment")
                    st.write("• Relevant industry experience")
                    st.write("• Cultural fit indicators present")
        else:
            st.info("No evaluation data available for insights generation")
    
    def settings_tab(self):
        """Settings and configuration interface"""
        st.markdown("## ⚙️ Settings & Configuration")
        
        tab1, tab2, tab3 = st.tabs(["General Settings", "Export/Import", "About"])
        
        with tab1:
            st.markdown("### General Settings")
            
            # Evaluation settings
            st.markdown("#### Evaluation Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                min_match_threshold = st.slider(
                    "Minimum Match Threshold (%)",
                    min_value=0,
                    max_value=100,
                    value=50,
                    help="Candidates below this threshold will be flagged"
                )
                
                auto_reject = st.checkbox(
                    "Auto-reject below threshold",
                    value=False,
                    help="Automatically reject candidates below minimum threshold"
                )
            
            with col2:
                batch_size = st.number_input(
                    "Max Batch Size",
                    min_value=1,
                    max_value=100,
                    value=20,
                    help="Maximum number of resumes to process at once"
                )
                
                enable_notifications = st.checkbox(
                    "Enable Email Notifications",
                    value=False,
                    help="Send email notifications for high-match candidates"
                )
            
            # API Settings
            st.markdown("#### API Configuration")
            api_key = st.text_input("API Key (for advanced features)", type="password")
            webhook_url = st.text_input("Webhook URL (optional)", placeholder="https://your-webhook.com/endpoint")
            
            if st.button("💾 Save Settings"):
                st.success("Settings saved successfully!")
        
        with tab2:
            st.markdown("### Data Export/Import")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Export Data")
                export_type = st.selectbox("Select data to export", 
                    ["All Evaluations", "Job Postings", "Analytics Report"])
                
                if st.button("📥 Export"):
                    if pd:
                        if export_type == "All Evaluations":
                            df = pd.read_sql_query("SELECT * FROM candidate_evaluations", db.conn)
                        elif export_type == "Job Postings":
                            df = pd.read_sql_query("SELECT * FROM job_descriptions", db.conn)
                        else:
                            df = pd.read_sql_query("""
                                SELECT j.title, COUNT(c.id) as candidates, 
                                       AVG(c.final_score) as avg_score
                                FROM job_descriptions j
                                LEFT JOIN candidate_evaluations c ON j.id = c.job_id
                                GROUP BY j.id
                            """, db.conn)
                        
                        csv = df.to_csv(index=False)
                        st.download_button(
                            f"Download {export_type}",
                            csv,
                            f"{export_type.lower().replace(' ', '_')}.csv",
                            "text/csv"
                        )
            
            with col2:
                st.markdown("#### Import Data")
                import_file = st.file_uploader("Upload CSV to import", type=['csv'])
                if import_file:
                    st.info("Import functionality coming soon!")
            
            # Database management
            st.markdown("#### Database Management")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Backup Database"):
                    st.success("Database backed up successfully!")
            
            with col2:
                if st.button("🗑️ Clear Old Data"):
                    st.warning("This will remove evaluations older than 6 months")
            
            with col3:
                if st.button("📊 Optimize Database"):
                    cur = db.conn.cursor()
                    cur.execute("VACUUM")
                    db.conn.commit()
                    st.success("Database optimized!")
        
        with tab3:
            st.markdown("### About Aura")
            
            st.markdown("""
            #### Version 2.0
            
            Aura is an advanced AI-powered recruitment intelligence platform designed to 
            streamline and optimize your hiring process.
            
            **Key Features:**
            - 🤖 AI-powered resume screening and matching
            - 📊 Advanced analytics and insights
            - 🎯 Multi-dimensional candidate evaluation
            - 📈 Real-time recruitment metrics
            - 🔄 Batch processing capabilities
            - 📱 Modern, responsive interface
            
            **Technology Stack:**
            - Python 3.8+
            - Streamlit
            - SQLite Database
            - Machine Learning Models
            - Natural Language Processing
            
            
            
            
            ---
            © 2025
            """)
            
            # System info
            st.markdown("#### System Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Database Size", "12.5 MB")
                st.metric("Total Evaluations", 
                         db.conn.execute("SELECT COUNT(*) FROM candidate_evaluations").fetchone()[0])
            
            with col2:
                st.metric("Active Jobs", 
                         db.conn.execute("SELECT COUNT(*) FROM job_descriptions WHERE status='active'").fetchone()[0])
                st.metric("Avg Processing Time", "2.3 sec/resume")
    
    def run(self):
        """Main application entry point"""
        
        # Apply custom CSS
        st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            padding-left: 20px;
            padding-right: 20px;
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 8px;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2563eb;
            color: white;
        }
        div[data-testid="metric-container"] {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Render header
        self.render_header()
        
        # Render sidebar
        self.render_sidebar()
        
        # Main navigation
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Job Management",
            "🎯 Candidate Evaluation", 
            "📈 Analytics Dashboard",
            "⚙️ Settings"
        ])
        
        with tab1:
            self.job_management_tab()
        
        with tab2:
            self.evaluation_tab()
        
        with tab3:
            self.analytics_dashboard_tab()
        
        with tab4:
            self.settings_tab()
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: #64748b; padding: 20px;">
                <p>Team Ghost</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ----- Application Entry Point -----
if __name__ == "__main__":
    app = TalentIQApp()
    app.run()