

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()




def _first(*names: str) -> str | None:
    """Verilen env adlarından ilk dolu olanı döndürür."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


@dataclass(frozen=True)
class Settings:
    # provider: "openai" → OpenAI-uyumlu (base_url sağlayıcıyı seçer). Diğerleri
    # (google_genai, anthropic, groq, ...) init_chat_model ile native kurulur.
    provider: str = field(default_factory=lambda: (os.getenv("LLM_PROVIDER") or "openai").strip())
    model_name: str = field(default_factory=lambda: _first("LLM_MODEL", "HF_MODEL") or "")
    # Boş bırakılırsa ChatOpenAI kendi varsayılanına (api.openai.com) düşer.
    base_url: str | None = field(default_factory=lambda: _first("LLM_BASE_URL", "HF_BASE_URL"))
    api_key: str | None = field(
        default_factory=lambda: _first(
            "LLM_API_KEY", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN",
            "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
            "ANTHROPIC_API_KEY", "GROQ_API_KEY",
        )
    )
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.1")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "4096")))
    # Tek bir LLM çağrısı için istek zaman aşımı (sn) + yeniden deneme. Sağlayıcı
    # bir bağlantıyı askıya alırsa çağrı SONSUZA KADAR beklemesin diye şart:
    # timeout dolunca hata verir, birkaç kez dener, olmazsa vaka error olur ve koşu
    # sıradaki soruya geçer (bloklanmaz).
    timeout: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT", "90")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "2")))
    # structured output yöntemi: boş → LangChain varsayılanı (OpenAI/HF'de json_schema).
    structured_method: str | None = field(default_factory=lambda: (os.getenv("LLM_STRUCTURED_METHOD") or "").strip() or None)
    # Triyaj modeli. Triyaj bir SINIFLANDIRMA işidir (hangi şerit + hangi araçlar) ve
    # her isteğin önünde durduğu için gecikmeye doğrudan etki eder; ayrıca `direct`
    # şeridinde cevabı da o yazar. Boş bırakılırsa ana model kullanılır.
    triage_model: str | None = field(default_factory=lambda: (os.getenv("TRIAGE_MODEL") or "").strip() or None)


settings = Settings()


def get_triage_llm():
    """Triyaj katmanının modeli — ayarlıysa TRIAGE_MODEL, değilse ana model."""
    if settings.triage_model:
        return get_llm(model=settings.triage_model)
    return get_llm()


def structured_output_kwargs() -> dict:
    """with_structured_output için ek argümanlar (yöntem ayarlıysa)."""
    return {"method": settings.structured_method} if settings.structured_method else {}


def get_llm(**overrides):
    """Konfigüre edilmiş bir sohbet modeli döndürür (sağlayıcıdan bağımsız).

    provider="openai" (varsayılan) ise ChatOpenAI + base_url kullanılır — bu tek
    yol OpenAI, Gemini, Groq, Together, OpenRouter, DeepInfra, HF Router ve yerel
    (Ollama/vLLM) sağlayıcılarını kapsar. Aksi halde init_chat_model ile native
    sağlayıcı kurulur (paketi kurulu olmalı).
    """
    if not settings.api_key:
        raise RuntimeError(
            "LLM API anahtarı bulunamadı. .env'e LLM_API_KEY=<anahtar> ekle "
            "(HF için HF_TOKEN da olur)."
        )
    if not settings.model_name:
        raise RuntimeError(
            "Model adı tanımlı değil. .env'e LLM_MODEL=<model> ekle "
            "(ör. gpt-4o, gemini-2.5-flash, Qwen/Qwen3.5-122B-A10B:deepinfra)."
        )

    # Azure OpenAI: standart OpenAI'den ayrı (deployment + api-version + api-key
    # header). AzureChatOpenAI kullanılır. Gereken env: AZURE_OPENAI_ENDPOINT,
    # AZURE_OPENAI_API_KEY, OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT (ya da LLM_MODEL).
    if settings.provider in ("azure", "azure_openai"):
        from langchain_openai import AzureChatOpenAI

        aparams = dict(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT") or settings.model_name,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT") or settings.base_url,
            api_version=os.getenv("OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21",
            api_key=settings.api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            timeout=settings.timeout,
            max_retries=settings.max_retries,
        )
        aparams.update(overrides)
        return AzureChatOpenAI(**aparams)

    params = dict(
        model=settings.model_name,
        api_key=settings.api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,          # tek çağrı zaman aşımı (askıyı önler)
        max_retries=settings.max_retries,  # geçici hatada yeniden dene
    )

    if settings.provider == "openai":
        from langchain_openai import ChatOpenAI

        if settings.base_url:
            params["base_url"] = settings.base_url
        params.update(overrides)
        return ChatOpenAI(**params)

    # Native sağlayıcı (google_genai, anthropic, groq, mistralai, ollama, ...).
    try:
        from langchain.chat_models import init_chat_model
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Native sağlayıcı için 'langchain' paketi gerekli: uv add langchain"
        ) from exc

    params.update(overrides)
    model = params.pop("model")
    try:
        return init_chat_model(model, model_provider=settings.provider, **params)
    except ImportError as exc:
        raise RuntimeError(
            f"'{settings.provider}' sağlayıcısının LangChain entegrasyonu kurulu değil. "
            f"Örn: uv add langchain-{settings.provider.replace('_', '-')}"
        ) from exc
