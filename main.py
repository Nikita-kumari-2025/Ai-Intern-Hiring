from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI(title="SHL Assessment Recommender API")


assessments = pd.DataFrame([
    {
        "title": "Numerical Reasoning Test",
        "url": "https://www.shl.com/en/assessments/numerical-reasoning/",
        "test_type": "Cognitive",
        "remote_testing": "Yes",
        "adaptive_support": "Yes",
        "duration_minutes": 20,
        "description": "Assesses numerical reasoning and data interpretation skills."
    },
    {
        "title": "Situational Judgment Test",
        "url": "https://www.shl.com/en/assessments/situational-judgement/",
        "test_type": "Behavioral",
        "remote_testing": "Yes",
        "adaptive_support": "No",
        "duration_minutes": 25,
        "description": "Evaluates decision making and judgment in work scenarios."
    },
    {
        "title": "Verbal Reasoning Test",
        "url": "https://www.shl.com/en/assessments/verbal-reasoning/",
        "test_type": "Cognitive",
        "remote_testing": "No",
        "adaptive_support": "Yes",
        "duration_minutes": 20,
        "description": "Measures verbal logic and reading comprehension."
    }
])

class Recommendation(BaseModel):
    title: str
    url: str
    test_type: str
    remote_testing: str
    adaptive_support: str
    duration_minutes: int

@app.get("/recommend", response_model=List[Recommendation])
def recommend(query: str = Query(..., description="Natural language query or job description"), top_n: int = 10):
    scored = assessments.copy()
    scored["score"] = 0.0

    for i, row in scored.iterrows():
        vec = CountVectorizer().fit_transform([row["description"], query])
        sim = cosine_similarity(vec)[0][1]
        scored.at[i, "score"] = sim

    recommended = scored.sort_values("score", ascending=False).head(top_n)
    recommended = recommended[recommended["score"] > 0]

    return recommended[[
        "title", "url", "test_type", "remote_testing", "adaptive_support", "duration_minutes"
    ]].to_dict(orient="records")