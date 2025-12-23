from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_prefix: str = "/api/v1"
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3])
    cors_allow_origins: List[str] = Field(default_factory=lambda: ["*"])
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_deployment: Optional[str] = None

    class Config:
        env_prefix = "CAIN_"
        env_file = ".env"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
