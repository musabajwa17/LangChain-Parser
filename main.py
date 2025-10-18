from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import parser  # import your parser router
from routers import parser, enrich

app = FastAPI(title="TaaS Grid Resume Parser API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(parser.router)
app.include_router(enrich.router)

@app.get("/")
def home():
    return {"message": "✅ TaaS Grid Backend is running properly"}
