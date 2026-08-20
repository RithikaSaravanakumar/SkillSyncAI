import json
import hashlib
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@dataclass
class Job:
    id: str
    title: str
    description: str
    requirements: dict
    created_at: str
    candidates: dict = field(default_factory=dict)


@dataclass
class Candidate:
    id: str
    filename: str
    resume_text: str
    analysis: dict | None = None
    content_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


JOBS = {}
RESUME_ANALYSIS_CACHE = {}
JOB_REQUIREMENTS_CACHE = {}
RESUME_TEXT_CACHE = {}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SKILL_WEIGHT = 0.40
EXPERIENCE_WEIGHT = 0.20
EDUCATION_WEIGHT = 0.10
PROJECT_WEIGHT = 0.10
CONTEXT_WEIGHT = 0.20

SKILL_DICTIONARY = {
    "programming": ["Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#", "Go", "PHP", "Ruby", "Kotlin", "Swift"],
    "frontend": ["HTML", "CSS", "React", "Angular", "Vue", "Next.js", "Tailwind CSS"],
    "backend": ["Node.js", "Express", "Spring", "Spring Boot", "Django", "Flask", "FastAPI"],
    "database": ["MySQL", "PostgreSQL", "MongoDB", "Oracle", "SQL Server", "Redis", "SQL"],
    "data_ai": ["Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Machine Learning", "Deep Learning", "NLP", "Computer Vision"],
    "cloud_devops": ["AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "Git", "GitHub", "CI/CD"],
    "analytics": ["Excel", "Power BI", "Tableau", "Statistics", "Data Analysis"],
}
SKILLS = [skill for category in SKILL_DICTIONARY.values() for skill in category]
SKILL_ALIASES = {"js": "javascript", "reactjs": "react", "react.js": "react", "nodejs": "node.js", "node": "node.js", "sklearn": "scikit-learn", "cicd": "ci/cd", "gcp": "google cloud"}


def _gemini_error_response(error):
    message = str(error)
    error_type = error.__class__.__name__
    if "ResourceExhausted" in error_type or "quota" in message.lower() or "429" in message:
        return jsonify({"error": "Gemini API quota is temporarily exhausted. Please wait and try again, or configure a Gemini project with available quota."}), 429
    if "Unauthenticated" in error_type or "401" in message or "API key" in message:
        return jsonify({"error": "Gemini API authentication is not configured correctly."}), 401
    return jsonify({"error": "Gemini API request failed. Check the backend logs for details."}), 502


def _gemini_error_message(error):
    message = str(error)
    error_type = error.__class__.__name__
    if "ResourceExhausted" in error_type or "quota" in message.lower() or "429" in message:
        return "Gemini API quota is temporarily exhausted. Please wait and try again."
    if "Unauthenticated" in error_type or "401" in message or "API key" in message:
        return "Gemini API authentication is not configured correctly."
    return "Gemini API request failed. Check the backend logs for details."

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        if text.strip():
            return text.strip()
    except Exception as e:
        print(f"Direct text extraction failed: {e}")

    print("Falling back to OCR for image-based PDF.")
    try:
        images = convert_from_path(pdf_path)
        for image in images:
            page_text = pytesseract.image_to_string(image)
            text += page_text + "\n"
    except Exception as e:
        print(f"OCR failed: {e}")

    extracted_text = text.strip()
    if not extracted_text:
        raise ValueError("No readable text could be extracted from this PDF. Use a text-based PDF or install OCR support.")
    return extracted_text

def _as_list(value):
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [line.strip(" -\t") for line in str(value).splitlines() if line.strip(" -\t")]


def _as_text(value, fallback="Not provided"):
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _as_score(value):
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(0, min(score, 100))


def _as_float(value, fallback=0.0):
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return fallback


def _normalise_skill(value):
    return re.sub(r"[^a-z0-9+#.]", "", str(value).lower())


def _unique_skills(values):
    result = []
    seen = set()
    for value in _as_list(values):
        key = _normalise_skill(value)
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def normalize_skill(value):
    text = re.sub(r"\s+", " ", str(value).strip().lower().replace("–", "-")).strip()
    text = text.replace("react js", "reactjs").replace("react.js", "reactjs").replace("node js", "nodejs")
    text = re.sub(r"[^a-z0-9+#./ -]", "", text)
    return SKILL_ALIASES.get(text, text)


def _skill_display_map():
    return {normalize_skill(skill): skill for skill in SKILLS}


def _extract_local_skills(text):
    lowered = text.lower()
    display_map = _skill_display_map()
    found = []
    for skill in SKILLS:
        pattern = re.escape(skill.lower()).replace(r"\.", r"\\.?" )
        if re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", lowered) or any(normalize_skill(alias) == normalize_skill(skill) and re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", lowered) for alias in SKILL_ALIASES if SKILL_ALIASES[alias] == normalize_skill(skill)):
            found.append(display_map[normalize_skill(skill)])
    normalized_found = _unique_skills(found)
    return [skill for skill in normalized_found if not any(skill != other and normalize_skill(skill) in normalize_skill(other) for other in normalized_found)]


def _section_text(text, headings):
    heading_pattern = "|".join(re.escape(heading) for heading in headings)
    match = re.search(rf"(?:^|\n)\s*(?:{heading_pattern})\s*:?\s*\n?(.*?)(?=\n\s*[A-Z][A-Za-z /&-]{{2,30}}\s*:?(?:\n|$)|\Z)", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_experience_years(text):
    matches = re.findall(r"(?<!\d)(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b", text.lower())
    return max((float(value) for value in matches), default=0.0)


def _extract_education(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    education_terms = ("b.e", "b.tech", "bachelor", "master", "m.e", "m.tech", "mca", "mba", "ph.d", "phd", "diploma", "computer science", "information technology")
    matches = [line for line in lines if any(term in line.lower() for term in education_terms)]
    return matches[0] if matches else ""


def _extract_name(text):
    for line in text.splitlines()[:12]:
        value = line.strip()
        if value and not re.search(r"@|https?://|\d{5,}|resume|curriculum vitae", value, re.IGNORECASE) and len(value.split()) <= 5:
            return value
    return ""


def _parse_local_resume(resume_text):
    skills = _extract_local_skills(resume_text)
    projects = _section_text(resume_text, ["projects", "academic projects", "project experience"])
    certifications = _section_text(resume_text, ["certifications", "certificates", "licenses"])
    education = _extract_education(resume_text)
    sections_found = sum(bool(_section_text(resume_text, [heading])) for heading in ["skills", "experience", "education", "projects", "certifications"])
    ats_score = min(100, len(skills) * 8 + sections_found * 8 + (10 if re.search(r"@|\+?\d[\d -]{8,}", resume_text) else 0))
    resume_score = min(100, ats_score + (10 if projects else 0) + (5 if certifications else 0))
    return {
        "candidateName": _extract_name(resume_text),
        "skills": skills,
        "experienceYears": _extract_experience_years(resume_text),
        "education": education,
        "projects": projects,
        "certifications": certifications,
        "atsScore": ats_score,
        "resumeScore": resume_score,
        "strengths": skills[:5],
        "weaknesses": [],
        "summary": " ".join(resume_text.split())[:500],
    }


def _parse_local_job(description, title=""):
    skills = _extract_local_skills(description)
    preferred_matches = re.findall(r"(?:preferred|nice to have|bonus|plus)\s*[:\-]?\s*(.*?)(?=\n|$)", description, re.IGNORECASE)
    preferred_text = " ".join(preferred_matches)
    preferred = [skill for skill in skills if skill.lower() in preferred_text.lower()]
    required = [skill for skill in skills if skill not in preferred]
    if not required and skills:
        required = skills
    return {"required_skills": required, "preferred_skills": preferred, "required_experience_years": _extract_experience_years(description), "required_education": _extract_education(description), "role_title": title or "Untitled role"}


def _safe_filename(filename):
    name = secure_filename(filename or "resume.pdf")
    return name or "resume.pdf"


def _analysis_timestamp():
    return datetime.now(timezone.utc).isoformat()


def _resume_hash(resume_text):
    return hashlib.sha256(resume_text.encode("utf-8")).hexdigest()


def _local_resume_profile(resume_text):
    cache_key = _resume_hash(resume_text)
    if cache_key not in RESUME_TEXT_CACHE:
        app.logger.info("[Ranking] Parsing unique resume locally")
        RESUME_TEXT_CACHE[cache_key] = _parse_local_resume(resume_text)
    return RESUME_TEXT_CACHE[cache_key].copy()


def _generate_content(prompt, purpose):
    model = genai.GenerativeModel(GEMINI_MODEL)
    for attempt in range(3):
        try:
            app.logger.info("[Gemini] %s", purpose)
            return model.generate_content(prompt)
        except Exception as error:
            message = str(error).lower()
            is_rate_limited = "resourceexhausted" in error.__class__.__name__.lower() or "429" in message or "quota" in message
            quota_exhausted = "generaterequestsperday" in message or "daily" in message
            if not is_rate_limited or quota_exhausted or attempt == 2:
                raise
            delay = 2 ** attempt
            app.logger.warning("[Gemini] rate limited during %s; retrying in %ss", purpose, delay)
            time.sleep(delay)


def _extract_json(text):
    text = text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        text = fenced_match.group(1).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
    return json.loads(text)


def _normalize_analysis(data):
    overview = data.get("candidate_overview", {})
    skill_gap = data.get("skill_gap_analysis", {})
    roadmap = data.get("actionable_roadmap", {})
    summary = data.get("hr_md_summary", {})
    ats = data.get("ats_insights", {})

    return {
        "candidate_overview": {
            "name": _as_text(overview.get("name")),
            "education": _as_text(overview.get("education")),
            "cgpa": _as_text(overview.get("cgpa")),
        },
        "skill_gap_analysis": {
            "current_skills": _as_list(skill_gap.get("current_skills")),
            "recommended_skills": _as_list(skill_gap.get("recommended_skills")),
        },
        "actionable_roadmap": {
            "courses": _as_list(roadmap.get("courses")),
            "project_ideas": _as_list(roadmap.get("project_ideas")),
        },
        "hr_md_summary": {
            "summary": _as_text(summary.get("summary")),
            "recommendation": _as_text(summary.get("recommendation"), "Train").title(),
        },
        "ats_insights": {
            "ats_score": _as_score(ats.get("ats_score")),
            "score_reason": _as_text(ats.get("score_reason")),
            "matched_keywords": _as_list(ats.get("matched_keywords")),
            "missing_keywords": _as_list(ats.get("missing_keywords")),
            "improvement_tips": _as_list(ats.get("improvement_tips")),
            "section_feedback": _as_list(ats.get("section_feedback")),
        },
    }


def analyze_resume(resume_text, job_description=None):
    if not resume_text:
        raise ValueError("Resume text is required for analysis.")
    
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    base_prompt = f"""
    You are an experienced HR and technical interviewer. Review the resume for roles such as Data Science, Data Analyst, DevOps, Machine Learning Engineer, Prompt Engineer, AI Engineer, Full Stack Web Development, Big Data Engineering, Marketing Analyst, HR Manager, or Software Developer.
    For ATS scoring, use a practical 0-100 score based on keyword match, role relevance, clear section headings, measurable project or internship impact, formatting simplicity, contact details, education, and certification evidence. If a job description is supplied, weigh its required keywords heavily. If no job description is supplied, infer the most likely entry-level target role from the resume.

    Return ONLY valid JSON with this exact schema and no Markdown:
    {{
      "candidate_overview": {{
        "name": "Candidate name if present",
        "education": "Degree, department, institution if present",
        "cgpa": "CGPA or percentage if present"
      }},
      "skill_gap_analysis": {{
        "current_skills": ["skill from resume"],
        "recommended_skills": ["skill to add or improve"]
      }},
      "actionable_roadmap": {{
        "courses": ["specific course or learning topic"],
        "project_ideas": ["specific project idea"]
      }},
      "hr_md_summary": {{
        "summary": "Executive summary in simple language for HR/MD",
        "recommendation": "Hire, Intern, or Train"
      }},
      "ats_insights": {{
        "ats_score": 0,
        "score_reason": "One short reason for the ATS score",
        "matched_keywords": ["important keyword already present"],
        "missing_keywords": ["important keyword to add based on the job description or target role"],
        "improvement_tips": ["specific ATS improvement tip"],
        "section_feedback": ["specific feedback for resume sections such as summary, skills, projects, internships, education, certifications, formatting"]
      }}
    }}

    Resume:
    {resume_text}
    """

    if job_description:
        base_prompt += f"""
        Additionally, compare this resume to the following job description:
        
        Job Description:
        {job_description}
        
        Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
        """

    response = model.generate_content(base_prompt)
    return _normalize_analysis(_extract_json(response.text))


def _extract_job_requirements(description, title=""):
    cache_key = hashlib.sha256(description.strip().encode("utf-8")).hexdigest()
    if cache_key in JOB_REQUIREMENTS_CACHE:
        app.logger.info("[Gemini] Reusing cached Job Description requirements")
        return JOB_REQUIREMENTS_CACHE[cache_key].copy()
    prompt = f"""
    Extract structured hiring requirements from this job description. Return ONLY valid JSON.
    Schema: {{"required_skills": ["skill"], "preferred_skills": ["skill"], "required_experience_years": 0, "required_education": "qualification", "role_title": "title"}}
    Separate must-have skills from nice-to-have skills. Infer experience years only when stated; otherwise use 0.
    Role title: {title}
    Job description: {description}
    """
    response = _generate_content(prompt, "Processing Job Description")
    data = _extract_json(response.text)
    if not isinstance(data, dict) or not isinstance(data.get("required_skills", []), list) or not isinstance(data.get("preferred_skills", []), list):
        raise ValueError("Gemini returned an invalid job requirement object.")
    requirements = {
        "required_skills": _unique_skills(data.get("required_skills")),
        "preferred_skills": _unique_skills(data.get("preferred_skills")),
        "required_experience_years": _as_float(data.get("required_experience_years")),
        "required_education": _as_text(data.get("required_education"), "Not specified"),
        "role_title": _as_text(data.get("role_title"), title or "Untitled role"),
    }
    JOB_REQUIREMENTS_CACHE[cache_key] = requirements
    return requirements.copy()


def _extract_candidate_profile(resume_text, label="Resume"):
    prompt = f"""
    Extract the candidate profile once for reuse by resume analysis and job matching. Return ONLY valid JSON:
    {{"candidateName":"Name or Not provided","skills":["skill"],"experienceYears":0,"education":"qualification or Not provided","atsScore":0,"resumeScore":0,"strengths":["strength"],"weaknesses":["weakness"],"summary":"concise factual summary"}}
    Do not invent values. Use 0 only when a score cannot be determined and "Not provided" when information is absent.
    Resume: {resume_text}
    """
    response = _generate_content(prompt, f"Analyzing {label}")
    data = _extract_json(response.text)
    required_fields = {"candidateName", "skills", "experienceYears", "education", "atsScore", "resumeScore", "strengths", "weaknesses", "summary"}
    if not isinstance(data, dict) or not required_fields.issubset(data) or not isinstance(data.get("skills"), list):
        raise ValueError("Gemini returned an invalid candidate object.")
    if not isinstance(data.get("experienceYears"), (int, float)) or isinstance(data.get("experienceYears"), bool):
        raise ValueError("Gemini returned an invalid experience value.")
    for score_field in ("atsScore", "resumeScore"):
        if not isinstance(data.get(score_field), (int, float)) or isinstance(data.get(score_field), bool) or not 0 <= float(data[score_field]) <= 100:
            raise ValueError(f"Gemini returned an invalid {score_field} value.")
    skills = _unique_skills(data.get("skills"))
    return {
        "candidateName": _as_text(data.get("candidateName")),
        "skills": skills,
        "experienceYears": _as_float(data.get("experienceYears")),
        "education": _as_text(data.get("education")),
        "atsScore": _as_score(data.get("atsScore")),
        "resumeScore": _as_score(data.get("resumeScore")),
        "strengths": _as_list(data.get("strengths")),
        "weaknesses": _as_list(data.get("weaknesses")),
        "summary": _as_text(data.get("summary"), "No candidate summary returned."),
    }


def _get_cached_profile(resume_text, label="Resume"):
    cache_key = _resume_hash(resume_text)
    if cache_key in RESUME_ANALYSIS_CACHE:
        app.logger.info("[Gemini] Reusing cached resume analysis %s", cache_key[:12])
        return RESUME_ANALYSIS_CACHE[cache_key].copy()
    profile = _extract_candidate_profile(resume_text, label)
    RESUME_ANALYSIS_CACHE[cache_key] = profile.copy()
    return profile


def _score_candidate(candidate, job):
    extracted = candidate.analysis
    requirements = job.requirements
    candidate_skills = {_normalise_skill(skill) for skill in extracted["skills"]}
    required = requirements["required_skills"]
    preferred = requirements["preferred_skills"]
    matching_required = [skill for skill in required if _normalise_skill(skill) in candidate_skills]
    matching_preferred = [skill for skill in preferred if _normalise_skill(skill) in candidate_skills]
    missing_required = [skill for skill in required if _normalise_skill(skill) not in candidate_skills]
    required_score = (len(matching_required) / len(required) * 100) if required else 100
    preferred_score = (len(matching_preferred) / len(preferred) * 100) if preferred else 100
    needed_experience = requirements["required_experience_years"]
    actual_experience = extracted["experienceYears"]
    experience_score = 100 if not needed_experience else min(actual_experience / needed_experience * 100, 100)
    required_education = _normalise_skill(requirements["required_education"])
    education_text = _normalise_skill(extracted["education"])
    education_score = 100 if required_education in ("", "notspecified") or required_education in education_text else 0
    resume_terms = set(re.findall(r"[a-z0-9+#.]+", extracted["summary"].lower()))
    skill_terms = {_normalise_skill(skill) for skill in extracted["skills"]}
    required_terms = {_normalise_skill(skill) for skill in requirements["required_skills"] + requirements["preferred_skills"]}
    semantic_score = round(len((resume_terms | skill_terms) & required_terms) / len(required_terms) * 100, 2) if required_terms else 0
    app.logger.info("[Ranking] Calculating scores locally")
    app.logger.info("[Ranking] No additional Gemini request required")
    final_score = round(required_score * .40 + preferred_score * .20 + experience_score * .20 + education_score * .10 + semantic_score * .10, 2)
    status = "EXCELLENT MATCH" if final_score >= 90 else "STRONG MATCH" if final_score >= 80 else "MODERATE MATCH" if final_score >= 70 else "LOW MATCH"
    extracted.update({
        "matchingSkills": matching_required,
        "missingSkills": missing_required,
        "preferredSkillsMatched": matching_preferred,
        "requiredSkillsScore": round(required_score, 2),
        "preferredSkillsScore": round(preferred_score, 2),
        "experienceMatch": round(experience_score, 2),
        "educationMatch": round(education_score, 2),
        "semanticSimilarity": semantic_score,
        "finalScore": final_score,
        "status": status,
        "scoreBreakdown": {"requiredSkills": round(required_score, 2), "preferredSkills": round(preferred_score, 2), "experience": round(experience_score, 2), "education": round(education_score, 2), "semanticSimilarity": semantic_score},
        "analysisTimestamp": _analysis_timestamp(),
        "explanation": f"Matched {len(matching_required)} of {len(required)} required skills and {len(matching_preferred)} of {len(preferred)} preferred skills. Experience score: {round(experience_score)}%; education score: {round(education_score)}%.",
    })
    return extracted


def _job_payload(job):
    return {"id": job.id, "title": job.title, "description": job.description, "requirements": job.requirements, "createdAt": job.created_at}


def _candidate_payload(candidate):
    return {"id": candidate.id, "filename": candidate.filename, "createdAt": candidate.created_at, "analysis": candidate.analysis}


def _ranked_candidates(job):
    candidates = [candidate for candidate in job.candidates.values() if candidate.analysis]
    candidates.sort(key=lambda item: (-item.analysis["finalScore"], item.analysis["candidateName"].lower(), item.id))
    return [{**_candidate_payload(candidate), "rank": index + 1} for index, candidate in enumerate(candidates)]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    start_time = time.perf_counter()
    uploaded_file = request.files.get('resume_file')
    job_description = request.form.get('job_description')

    if not uploaded_file or uploaded_file.filename == '':
        return jsonify({"error": "Please upload a resume in PDF format."}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    uploaded_file.save(filepath)
    resume_text = extract_text_from_pdf(filepath)

    try:
        analysis = analyze_resume(resume_text, job_description)
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {e}"}), 500

    total_time = round(time.perf_counter() - start_time, 2)
    return jsonify({"analysis": analysis, "total_time": total_time})


@app.route('/api/ranking/jobs', methods=['POST'])
def create_ranking_job():
    payload = request.get_json(silent=True) or {}
    title = str(payload.get('title') or request.form.get('title') or '').strip()
    description = str(payload.get('description') or payload.get('job_description') or request.form.get('description') or '').strip()
    job_file = request.files.get('job_description_file')
    if job_file and job_file.filename:
        filename = _safe_filename(job_file.filename)
        if not filename.lower().endswith('.pdf'):
            return jsonify({"error": "Job descriptions must be PDF files when uploaded."}), 400
        if job_file.content_length and job_file.content_length > MAX_UPLOAD_BYTES:
            return jsonify({"error": "Job description files must be 10MB or smaller."}), 400
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"job_{uuid.uuid4()}_{filename}")
        try:
            job_file.save(filepath)
            if os.path.getsize(filepath) > MAX_UPLOAD_BYTES:
                return jsonify({"error": "Job description files must be 10MB or smaller."}), 400
            description = extract_text_from_pdf(filepath).strip()
        except Exception:
            return jsonify({"error": "The uploaded job description could not be processed."}), 400
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    app.logger.info("Ranking job description received: length=%d source=%s", len(description), "pdf" if job_file and job_file.filename else "text")
    if len(description) < 30:
        return jsonify({"error": "Please provide a job description with enough detail to match candidates."}), 400
    cache_key = hashlib.sha256(description.strip().encode("utf-8")).hexdigest()
    if cache_key in JOB_REQUIREMENTS_CACHE:
        app.logger.info("[Ranking] Reusing cached local Job Description requirements")
        requirements = JOB_REQUIREMENTS_CACHE[cache_key].copy()
    else:
        app.logger.info("[Ranking] Extracting JD skills")
        requirements = _parse_local_job(description, title)
        JOB_REQUIREMENTS_CACHE[cache_key] = requirements.copy()
    existing_job = next((candidate for candidate in JOBS.values() if candidate.title == requirements["role_title"] and candidate.description == description), None)
    if existing_job:
        app.logger.info("[Ranking] Reusing existing candidate collection: %d candidates", len(existing_job.candidates))
        return jsonify({"job": _job_payload(existing_job)}), 200
    job = Job(str(uuid.uuid4()), requirements["role_title"], description, requirements, _analysis_timestamp())
    JOBS[job.id] = job
    return jsonify({"job": _job_payload(job)}), 201


@app.route('/api/ranking/jobs/<job_id>', methods=['DELETE'])
def clear_ranking_job(job_id):
    if job_id not in JOBS:
        return jsonify({"error": "Ranking session not found."}), 404
    JOBS.pop(job_id, None)
    app.logger.info("[Ranking] Candidate collection cleared")
    return jsonify({"success": True}), 200


def _get_job(job_id):
    job = JOBS.get(job_id)
    if not job:
        return None, (jsonify({"error": "Job not found."}), 404)
    return job, None


@app.route('/api/ranking/jobs/<job_id>/candidates', methods=['POST'])
def add_ranking_candidate(job_id):
    job, error_response = _get_job(job_id)
    if error_response:
        return error_response
    uploaded_files = request.files.getlist('resumes') or request.files.getlist('resume_file')
    uploaded_files = [uploaded_file for uploaded_file in uploaded_files if uploaded_file and uploaded_file.filename]
    app.logger.info("[Ranking] Received resumes: %d", len(uploaded_files))
    if not uploaded_files:
        return jsonify({"error": "Please upload at least one candidate PDF resume."}), 400
    added = []
    failures = []
    app.logger.info("[Ranking] Existing candidates: %d", len(job.candidates))
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        filename = _safe_filename(uploaded_file.filename)
        app.logger.info("[Ranking] Processing Resume %d/%d", index, len(uploaded_files))
        if not filename.lower().endswith('.pdf'):
            failures.append({"filename": filename, "error": "Candidate resumes must be PDF files."})
            continue
        if any(candidate.filename.lower() == filename.lower() for candidate in job.candidates.values()):
            failures.append({"filename": filename, "error": "This candidate file has already been added."})
            continue
        candidate_id = str(uuid.uuid4())
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{candidate_id}_{filename}")
        try:
            uploaded_file.save(filepath)
            if os.path.getsize(filepath) > MAX_UPLOAD_BYTES:
                raise ValueError("Candidate resumes must be 10MB or smaller.")
            resume_text = extract_text_from_pdf(filepath)
            if not resume_text.strip():
                raise ValueError("No readable text could be extracted from this resume.")
            if any(candidate.content_hash == _resume_hash(resume_text) for candidate in job.candidates.values()):
                raise ValueError("This resume has already been added to the ranking session.")
            candidate = Candidate(candidate_id, filename, resume_text, content_hash=_resume_hash(resume_text))
            job.candidates[candidate.id] = candidate
            added.append(_candidate_payload(candidate))
            app.logger.info("[Ranking] Candidate added: %s; total candidates: %d", filename, len(job.candidates))
        except Exception as error:
            app.logger.warning("[Ranking] Resume %s failed: %s", filename, error)
            failures.append({"filename": filename, "error": "Unable to process this resume locally.", "detail": str(error) if app.debug else None})
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    if not added:
        return jsonify({"error": "No candidate resumes could be processed.", "candidates": [], "failures": failures}), 400
    return jsonify({"candidates": added, "failures": failures}), 201


@app.route('/api/ranking/jobs/<job_id>/analyze', methods=['POST'])
def analyze_ranking_candidates(job_id):
    job, error_response = _get_job(job_id)
    if error_response:
        return error_response
    if not job.candidates:
        return jsonify({"error": "Add at least one candidate resume before analyzing."}), 400
    failures = []
    total_candidates = len(job.candidates)
    for index, candidate in enumerate(job.candidates.values(), start=1):
        try:
            app.logger.info("[Ranking] Processing resume %d of %d: %s", index, total_candidates, candidate.filename)
            candidate.analysis = _local_resume_profile(candidate.resume_text)
            _score_candidate(candidate, job)
        except Exception as error:
            app.logger.exception("[Ranking] Local processing failed for %s: %s", candidate.filename, error)
            failures.append({"candidateId": candidate.id, "filename": candidate.filename, "error": "Unable to process this resume locally.", "detail": str(error) if app.debug else None})
    ranked = _ranked_candidates(job)
    if not ranked:
        return jsonify({"error": failures[0]["error"] if failures else "No candidate could be analyzed.", "failures": failures}), 400
    app.logger.info("[Ranking] Ranking completed: %d candidates", len(ranked))
    return jsonify({"job": _job_payload(job), "candidates": ranked, "failures": failures})


@app.route('/api/ranking/jobs/<job_id>/candidates', methods=['GET'])
def list_ranking_candidates(job_id):
    job, error_response = _get_job(job_id)
    if error_response:
        return error_response
    return jsonify({"candidates": _ranked_candidates(job)})


@app.route('/api/ranking/jobs/<job_id>/ranking', methods=['GET'])
def get_ranking(job_id):
    job, error_response = _get_job(job_id)
    if error_response:
        return error_response
    ranked = _ranked_candidates(job)
    scores = [candidate["analysis"]["finalScore"] for candidate in ranked]
    return jsonify({"job": _job_payload(job), "candidates": ranked, "summary": {"totalCandidates": len(ranked), "averageScore": round(sum(scores) / len(scores), 2) if scores else 0, "topCandidate": ranked[0]["analysis"]["candidateName"] if ranked else None}})


@app.route('/api/ranking/jobs/<job_id>/candidates/<candidate_id>', methods=['GET'])
def get_ranking_candidate(job_id, candidate_id):
    job, error_response = _get_job(job_id)
    if error_response:
        return error_response
    candidate = job.candidates.get(candidate_id)
    if not candidate:
        return jsonify({"error": "Candidate not found."}), 404
    ranked = _ranked_candidates(job)
    rank = next((item["rank"] for item in ranked if item["id"] == candidate_id), None)
    return jsonify({"candidate": {**_candidate_payload(candidate), "rank": rank}, "job": _job_payload(job)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

