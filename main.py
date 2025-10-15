import os
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

# Function to extract text from PDF
def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Prompt template for structured extraction
template = PromptTemplate(
    input_variables=["resume_text"],
    template="""
Extract the following structured information from the resume below:

- Full Name
- Email
- Phone Number
- Skills
- Education (degree, institution, year)
- Work Experience (role, company, years)

Return ONLY valid JSON format (no explanations, no extra text).

Resume Text:
{resume_text}
"""
)

# API endpoint
@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    try:
        # Save uploaded PDF temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Extract text from PDF
        resume_text = extract_text_from_pdf(temp_path)

        # Run LLM
        chain = template | llm
        structured_response = chain.invoke({"resume_text": resume_text}).content

        # Clean up JSON
        structured_response = structured_response.replace("```json", "").replace("```", "").strip()
        structured_data = json.loads(structured_response)

        # Optionally, convert to DataFrame for internal usage
        df = pd.DataFrame([structured_data])

        # Remove temp file
        os.remove(temp_path)

        return JSONResponse(content=structured_data)

    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "raw_response": structured_response if 'structured_response' in locals() else None},
            status_code=500
        )
