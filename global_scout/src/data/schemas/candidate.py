from pydantic import BaseModel, Field
from typing import Optional

class Candidate(BaseModel):
    player_id: str
    name: str
    age: int = Field(ge=14, le=45)
    role: str  # batter | bowler | allrounder | keeper
    league_level: int = Field(ge=1, le=5)  # 1=elite .. 5=local
    matches: int = Field(ge=0)

    batting_sr: Optional[float] = None
    batting_avg: Optional[float] = None
    bowling_eco: Optional[float] = None
    bowling_avg: Optional[float] = None
    fielding_eff: Optional[float] = None  # 0..1
    recent_form: float = Field(ge=0.0, le=1.0)

    region: str  # for fairness auditing only, not a boost
