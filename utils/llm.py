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

import json
import os
import time
from typing import Tuple

from utils.config import LLMProvider


def initialize_client(provider: LLMProvider, azure_api_version: str = "2024-02-15-preview"):
    """Create an LLM client based on provider type."""
    
    if provider == LLMProvider.AZURE_OPENAI:
        from openai import AzureOpenAI
        
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_KEY")
        
        if not endpoint or not api_key:
            raise ValueError(
                "Azure OpenAI needs: AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY"
            )
        
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=azure_api_version
        )
    
    elif provider == LLMProvider.CLAUDE:
        from anthropic import Anthropic
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Claude needs: ANTHROPIC_API_KEY")
        
        return Anthropic(api_key=api_key)
    
    elif provider == LLMProvider.GEMINI:
        from google.genai import Client
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Gemini needs: GOOGLE_API_KEY")
        
        return Client(api_key=api_key)
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


def call_llm(
    client,
    provider: LLMProvider,
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Tuple[int, float, str, str]:
    """Call LLM and get prediction.
    
    Returns: (label, confidence, explanation, decision)
    """
    
    system_msg = """You are a clinical decision support system. Predict disease outcome (0=no, 1=yes).

Respond as JSON:
{
    "prediction": <0 or 1>,
    "confidence": <0.0 to 1.0>,
    "explanation": "<brief reason>",
    "decision": "<predict or defer>"
}"""
    
    for attempt in range(max_retries):
        try:
            response_text = _get_llm_response(
                client, provider, model, system_msg, prompt, temperature
            )
            
            result = _parse_response(response_text)
            return result
        
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"LLM error after {max_retries} attempts: {e}")
                return 0, 50.0, "Error", "defer"
            
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(retry_delay)
    
    return 0, 50.0, "Max retries exceeded", "defer"


def _get_llm_response(
    client, provider: LLMProvider, model: str, system_msg: str, prompt: str, temp: float
) -> str:
    """Get response text from LLM provider."""
    
    if provider == LLMProvider.AZURE_OPENAI:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=temp,
            max_completion_tokens=500,
        )
        return response.choices[0].message.content
    
    elif provider == LLMProvider.CLAUDE:
        response = client.messages.create(
            model=model,
            max_tokens=500,
            temperature=temp,
            system=system_msg,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    
    elif provider == LLMProvider.GEMINI:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": temp, "system_instruction": system_msg}
        )
        # Extract only text parts, ignoring non-text content like citations
        return response.text
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _parse_response(response_text: str) -> Tuple[int, float, str, str]:
    """Parse JSON response from LLM."""
    
    # Remove markdown code blocks if present
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    
    # Parse JSON
    data = json.loads(text.strip())
    
    label = int(data.get("prediction", 0))
    confidence = float(data.get("confidence", 0.5)) * 100
    explanation = str(data.get("explanation", "No explanation"))
    decision = str(data.get("decision", "defer")).lower()
    
    if confidence < 50:
        decision = "defer"
    
    return label, confidence, explanation, decision
