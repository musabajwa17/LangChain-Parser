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
# PROMPT TEMPLATE
# ----------------------------
template = PromptTemplate(
    input_variables=["resume_text"],
    template="""
Extract the following structured information from the resume below:

- name
- email
- phone
- citations
- impactFactor
- scholar
- education (degree, institution, year)
- experience (role, institute, years)
- achievements
- bookAuthorship
- journalGuestEditor
- researchPublications (journal,workshop)
- bookChapters
- msStudentsSupervised
- phdStudentsSupervised
- researchProjects
- professionalServices
- professionalTraining (title, description,year)
- technicalSkilss (title, lang)
- membershipsAndAssociations (heading, desc)
- references (prof, designation,mail,phone)

Return ONLY valid JSON format (no explanations, no extra text).

Resume Text:
{resume_text}
"""
)

# ----------------------------
# PARSING ENDPOINT
# ----------------------------
@router.post("/employee-parser")
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

        if not resume_text.strip():
            os.remove(temp_path)
            return JSONResponse(content={"error": "No readable text found. Try uploading a text-based resume."}, status_code=400)

        resume_text = resume_text[:6000]
        chain = template | llm
        structured_response = chain.invoke({"resume_text": resume_text}).content
        structured_data = clean_json_output(structured_response)

        os.remove(temp_path)
        return JSONResponse(content=structured_data)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
