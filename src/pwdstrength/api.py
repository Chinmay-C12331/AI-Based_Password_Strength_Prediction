"""FastAPI microservice for password guessability scoring.

CRITICAL ETHICAL ENFORCEMENT (Project Rule 4):
- NEVER log passwords or request bodies.
- NEVER write query passwords to disk, memory queues, or databases.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import numpy as np
import math

from pwdstrength.features import extract_features, get_feature_reasons
from pwdstrength.baselines import CompositionBaseline

app = FastAPI(
    title="pwd-strength-ai scoring service",
    description="Privacy-preserving API endpoint for password guessability prediction",
    version="0.1.0",
)

# Global in-memory baseline
_COMP_BASELINE = CompositionBaseline()


class ScoreRequest(BaseModel):
    # Field without logging or echo
    password: str = Field(..., max_length=128, description="Password to evaluate")


class ScoreResponse(BaseModel):
    y_hat: float
    level: int
    level_name: str
    guesses_log10: float
    reasons: List[str]


@app.post("/score", response_model=ScoreResponse)
async def score_password(req: ScoreRequest):
    """Score a password's guessability without logging or storing the input."""
    pw = req.password
    if not pw:
        raise HTTPException(status_code=400, detail="Password cannot be empty")

    feats = extract_features(pw)
    reasons = get_feature_reasons(pw, feats)

    y_est = float(_COMP_BASELINE.predict_y([pw])[0])
    cls_est = int(_COMP_BASELINE.predict_class([pw])[0])
    level_names = ["Very Weak", "Weak", "Fair", "Strong", "Very Strong"]

    return ScoreResponse(
        y_hat=y_est,
        level=cls_est,
        level_name=level_names[cls_est],
        guesses_log10=y_est,
        reasons=reasons,
    )
