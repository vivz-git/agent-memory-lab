import os
import pytest
from src.agent.core import BaseLLMClient, OpenAILLMClient, GroqLLMClient, get_llm_client

def test_openai_client_initialization(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    
    client = OpenAILLMClient()
    assert client.api_key == "test-openai-key"

def test_groq_client_initialization(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    
    client = GroqLLMClient()
    assert client.api_key == "test-groq-key"
    assert client.model == "openai/gpt-oss-120b"

def test_get_llm_client_factory_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    
    client = get_llm_client()
    assert isinstance(client, OpenAILLMClient)
    assert client.api_key == "test-openai"

def test_get_llm_client_factory_groq(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    
    client = get_llm_client()
    assert isinstance(client, GroqLLMClient)
    assert client.api_key == "test-groq"
    assert client.model == "openai/gpt-oss-120b"
