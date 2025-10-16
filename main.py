import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
import json
import pandas as pd

# Load environment variables
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Initialize FastAPI
app = FastAPI(title="Resume Parser API")

# Enable CORS (very important for your frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Extract text from PDF
def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Prompt template for structured output
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
- projects (name, technologies, description, link)
- certifications (title)
- location
- github
- linedin
- title

Return ONLY valid JSON format (no explanations, no extra text).

Resume Text:
{resume_text}
"""
)

# ---------------------------
# ✅ Resume Parsing Endpoint
# ---------------------------
@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        resume_text = extract_text_from_pdf(temp_path)

        chain = template | llm
        structured_response = chain.invoke({"resume_text": resume_text}).content

        # Clean JSON
        structured_response = (
            structured_response.replace("```json", "")
            .replace("```", "")
            .strip()
        )
        structured_data = json.loads(structured_response)

        os.remove(temp_path)

        return JSONResponse(content=structured_data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Simple test route
@app.get("/")
def home():
    return {"message": "✅ Resume Parser API running"}
