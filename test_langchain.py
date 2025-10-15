# import os
# import json
# import pandas as pd
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain

# # Load environment variables
# load_dotenv()

# # Get API key safely
# api_key = os.getenv("OPENAI_API_KEY")

# if not api_key:
#     raise ValueError("❌ OPENAI_API_KEY not found. Make sure it's in your .env file.")

# # Set for LangChain (optional, for redundancy)
# os.environ["OPENAI_API_KEY"] = api_key


# # ✅ Corrected function
# def extract_text_from_pdf(pdf_path):
#     reader = PdfReader(pdf_path)
#     text = ""
#     for page in reader.pages:
#         page_text = page.extract_text()
#         if page_text:
#             text += page_text
#     return text


# # ✅ Load your resume
# resume_text = extract_text_from_pdf("Musa_Cv.pdf")
# print(resume_text[:50000])  # print first 1000 characters to confirm it’s reading


# # ✅ Prompt for structured extraction
# template = PromptTemplate(
#     input_variables=["resume_text"],
#     template="""
#     Extract the following information from the resume text:
#     - Full Name
#     - Email
#     - Phone Number
#     - Skills
#     - Education (degree, institution, year)
#     - Work Experience (role, company, years)
#     Return the result strictly as a valid JSON object (no code block formatting).
#     Resume Text:
#     {resume_text}
#     """,
# )

# # ✅ LLM setup
# llm = ChatOpenAI(temperature=0, model_name="gpt-4.1-mini")

# # ✅ Chain
# chain = LLMChain(llm=llm, prompt=template)

# structured_response = chain.run(resume_text=resume_text)
# structured_response = structured_response.strip().replace("```", "").replace("json", "")

# print("🔹 Raw Response:")
# print(structured_response)


# # ✅ Convert JSON safely
# try:
#     structured_data = json.loads(structured_response)
#     df = pd.DataFrame([structured_data])
#     print("\n✅ Extracted Resume Information:")
#     print(df)
# except Exception as e:
#     print("❌ Error parsing JSON:", e)


import os
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

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Load your resume
resume_text = extract_text_from_pdf("Musa_Cv.pdf")

# Define prompt
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

# Run the chain
chain = template | llm

print("Processing your resume... please wait.\n")

structured_response = chain.invoke({"resume_text": resume_text}).content

# Clean up text and parse JSON
structured_response = structured_response.replace("```json", "").replace("```", "").strip()

try:
    structured_data = json.loads(structured_response)
    df = pd.DataFrame([structured_data])
    print("\n✅ Extracted Resume Data:\n")
    print(df)
except Exception as e:
    print("❌ JSON parsing error:", e)
    print("\nRaw response:\n", structured_response)
