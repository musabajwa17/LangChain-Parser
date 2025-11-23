import os
import json
import re
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from docx import Document
import pytesseract
from dotenv import load_dotenv

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

router = APIRouter()

# ----------------------------
# Initialize LLM
# ----------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ----------------------------
# PDF TEXT EXTRACTION (with OCR fallback)
# ----------------------------
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except Exception:
        pass

    if not text.strip():
        try:
            images = convert_from_path(file_path)
            ocr_text = "\n".join(pytesseract.image_to_string(img) for img in images)
            text += ocr_text
        except Exception as e:
            print("OCR failed:", e)
    return text.strip()

# ----------------------------
# DOCX TEXT EXTRACTION
# ----------------------------
def extract_text_from_docx(file_path: str) -> str:
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print("DOCX extraction failed:", e)
        return ""

# ----------------------------
# CLEAN JSON OUTPUT
# ----------------------------
def clean_json_output(text: str):
    text = text.replace("```json", "").replace("```", "").strip()
    text = re.sub(r'^[^{]*', '', text)
    text = re.sub(r'[^}]*$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON output from LLM", "raw_output": text}

# ----------------------------
# NORMALIZE RESUME DATA
# ----------------------------
def normalize_resume(data: dict):
    def normalize_list_of_strings(arr, key="name"):
        if isinstance(arr, list):
            # Deduplicate and convert strings → objects
            unique = list(dict.fromkeys(arr))
            return [{key: item} if isinstance(item, str) else item for item in unique]
        return []

    # Ensure keys exist
    keys_with_array_objects = ["certifications", "education", "experience", "projects"]
    for key in keys_with_array_objects:
        if key not in data or not isinstance(data[key], list):
            data[key] = []

    # Normalize certifications
    data["certifications"] = normalize_list_of_strings(data.get("certifications", []), key="name")

    # Normalize education
    normalized_edu = []
    for edu in data.get("education", []):
        if isinstance(edu, str):
            normalized_edu.append({"degree": edu, "institution": "", "year": ""})
        else:
            normalized_edu.append({
                "degree": edu.get("degree", ""),
                "institution": edu.get("institution", ""),
                "year": edu.get("year", "")
            })
    data["education"] = normalized_edu

    # Normalize experience
    normalized_exp = []
    for exp in data.get("experience", []):
        if isinstance(exp, str):
            normalized_exp.append({"role": exp, "company": "", "years": ""})
        else:
            normalized_exp.append({
                "role": exp.get("role", ""),
                "company": exp.get("company", ""),
                "years": exp.get("years", "")
            })
    data["experience"] = normalized_exp

    # Normalize projects
    normalized_proj = []
    for proj in data.get("projects", []):
        if isinstance(proj, str):
            normalized_proj.append({"name": proj, "domain": "", "description": "", "link": ""})
        else:
            normalized_proj.append({
                "name": proj.get("name", ""),
                "domain": proj.get("domain", ""),
                "description": proj.get("description", ""),
                "link": proj.get("link", "")
            })
    data["projects"] = normalized_proj

    # Skills → ensure array of strings
    if not isinstance(data.get("skills"), list):
        data["skills"] = []

    # Other optional fields
    optional_fields = ["name", "email", "phone", "summary", "location", "github", "linkedin", "title"]
    for field in optional_fields:
        if field not in data:
            data[field] = ""

    return data

# ----------------------------
# PROMPT TEMPLATE
# ----------------------------
template = PromptTemplate(
    input_variables=["resume_text"],
    template="""
Extract the following structured information from the resume below:

- name
- email
- phone
- skills
- summary
- education (degree, institution, year)
- experience (role, company, years)
- projects (name, domain, description, link)
- certifications (name)
- location
- github
- linkedin
- title

Return ONLY valid JSON format (no explanations, no extra text).

Resume Text:
{resume_text}
"""
)

# ----------------------------
# PARSING ENDPOINT
# ----------------------------
@router.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    try:
        if not (file.filename.lower().endswith(".pdf") or file.filename.lower().endswith(".docx")):
            return JSONResponse(content={"error": "Unsupported file type. Please upload PDF or DOCX only."}, status_code=400)

        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        if file.filename.lower().endswith(".pdf"):
            resume_text = extract_text_from_pdf(temp_path)
        else:
            resume_text = extract_text_from_docx(temp_path)

        os.remove(temp_path)

        if not resume_text.strip():
            return JSONResponse(content={"error": "No readable text found. Try uploading a text-based resume."}, status_code=400)

        resume_text = resume_text[:6000]  # limit for LLM
        chain = template | llm
        structured_response = chain.invoke({"resume_text": resume_text}).content
        raw_data = clean_json_output(structured_response)
        structured_data = normalize_resume(raw_data)

        return JSONResponse(content=structured_data)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)