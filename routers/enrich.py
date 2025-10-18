import os
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()
router = APIRouter()

# ----------------------------
# Initialize LangChain model
# ----------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ----------------------------
# Input Schema
# ----------------------------
class EnrichRequest(BaseModel):
    parsed_data: dict
    selected_fields: dict  # {"role": "frontend-developer", "industry": "technology", "experience_level": "mid-level", "tone": "formal"}

# ----------------------------
# Smarter Prompt Template
# ----------------------------
template = PromptTemplate(
    input_variables=["parsed_data", "selected_fields"],
    template="""
You are a **professional resume analyst and recruiter assistant**.  
Your job is to carefully study the parsed resume (in JSON format) and the user's selected context — including *role, industry, experience level,* and *tone*.

⚙️ Your responsibilities:
- Identify **what’s missing** from the resume for the given context.
- Suggest **only relevant additions** (no unrelated domains).
- Do **not** suggest generic "learn X" advice. Only content that can appear **in the resume**.
- Be **role-specific**. Example:
  - If the role = "Frontend Developer", do NOT suggest backend tools.
  - If the role = "Backend Developer", focus on APIs, databases, scalability, etc.
  - If the industry = "Healthcare", mention compliance, patient data handling, etc.
  - If experience = "Entry Level", suggest academic/research projects, not management skills.
  - If experience = "Senior Level", emphasize leadership and achievements.

Return a **strict JSON object only** with:
{{
  "summary_improvement": "An improved or missing summary section (if applicable)",
  "missing_sections": ["Projects", "Certifications", ...],
  "missing_details": ["Add quantifiable achievements in your experience section", ...],
  "suggested_additions": ["Add project on building dashboard using React + REST APIs", ...],
  "tone_recommendation": "Formal/Technical/etc — how the tone should appear for this role"
}}

Parsed CV (JSON):
{parsed_data}

User Context:
{selected_fields}
"""
)

# ----------------------------
# API Endpoint
# ----------------------------
@router.post("/enrich")
async def enrich_cv(request: EnrichRequest):
    try:
        chain = template | llm
        response = chain.invoke({
            "parsed_data": json.dumps(request.parsed_data, indent=2),
            "selected_fields": json.dumps(request.selected_fields, indent=2)
        })

        try:
            enriched = json.loads(response.content)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Model returned invalid JSON.")

        # Merge intelligently
        combined_data = {**request.parsed_data}

        # If user already has similar keys, enrich rather than overwrite
        combined_data["ai_enrichment"] = enriched

        return JSONResponse(content={
            "status": "success",
            "combined_cv": combined_data,
            "suggestions": enriched
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error enriching CV: {str(e)}")
