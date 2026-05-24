# Copyright (C) 2026 Panagiotis Tirchas
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv


# Load .env file at startup
load_dotenv()


class LLMProvider(Enum):
    """Supported LLM providers."""
    AZURE_OPENAI = "azure_openai"
    CLAUDE = "claude"
    GEMINI = "gemini"


@dataclass
class Config:
    """Experiment configuration."""
    llm_provider: LLMProvider
    llm_model: str
    num_perturbations: int
    num_stochastic_replicates: int
    random_seed: int
    max_retries: int
    retry_delay: float
    temperature: float
    azure_api_version: str
    use_retrieval: bool
    context_data_path: str
    tier: int = 1


def load_config() -> Config:
    """Load configuration from environment variables."""
    
    provider_str = os.getenv("LLM_PROVIDER", "azure_openai").lower()
    try:
        provider = LLMProvider(provider_str)
    except ValueError:
        print(f"Warning: Unknown LLM_PROVIDER '{provider_str}', using azure_openai")
        provider = LLMProvider.AZURE_OPENAI
    
    # Get model name based on provider
    if provider == LLMProvider.AZURE_OPENAI:
        model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4")
    elif provider == LLMProvider.CLAUDE:
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20250219")
    elif provider == LLMProvider.GEMINI:
        model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
    else:
        model = "unknown"

    return Config(
        llm_provider=provider,
        llm_model=model,
        num_perturbations=int(os.getenv("NUM_PERTURBATIONS", "2")),
        num_stochastic_replicates=int(os.getenv("NUM_STOCHASTIC_REPLICATES", "1")),
        random_seed=int(os.getenv("RANDOM_SEED", "0")),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        retry_delay=float(os.getenv("RETRY_DELAY", "1.0")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        azure_api_version=os.getenv("AZURE_API_VERSION", "2024-02-15-preview"),
        use_retrieval=os.getenv("USE_RETRIEVAL", "false").lower() in ("true", "1", "yes"),
        context_data_path=os.getenv("CONTEXT_DATA_PATH", "data/context_ready.csv"),
        tier=1,
    )


def print_config(config: Config) -> None:
    """Print configuration to console."""
    print("=" * 70)
    print("LLM Perturbation Stability Experiment")
    print("=" * 70)
    print(f"  Provider:        {config.llm_provider.value}")
    print(f"  Model:           {config.llm_model}")
    print(f"  Temperature:     {config.temperature}")
    print(f"  Perturbations:   {config.num_perturbations}")
    print(f"  Replicates:      {config.num_stochastic_replicates}")
    print(f"  Random seed:     {config.random_seed}")
    print(f"  Max retries:     {config.max_retries}")
    print(f"  Retry delay:     {config.retry_delay}s")
    print(f"  Use retrieval:   {config.use_retrieval}")
    if config.use_retrieval:
        print(f"  Context data:    {config.context_data_path}")
    print("=" * 70 + "\n")
