from pydantic import BaseModel, EmailStr
from typing import List, Literal

class EvaluationRequest(BaseModel):
    sequence: str
    email: EmailStr
    programs: List[Literal["mafft", "clustalo", "muscle", "t_coffee", "probcons", "kalign", "prank"]]

class EvalResult(BaseModel):
    tool: str
    blosum_score: float
    entropy: float
    gap_fraction: float
    cpu_time_sec: float
    memory_usage_mb: float

class MultiToolResult(BaseModel):
    session_id: str
    results: List[EvalResult]

class JobResponse(BaseModel):
    session_id: str
    message: str
    email: str
    programs: List[str]
    status: str
