import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:0.5b"

app = FastAPI(
    title="LLM Spam Detection Service",
    description="FastAPI wrapper around Ollama running Qwen2.5:0.5B for SMS spam classification.",
    version="1.0.0",
)


class GenerateRequest(BaseModel):
    """Request body for the /generate endpoint."""

    prompt: str = Field(..., description="User message / prompt text.")
    system: str = Field(
        default="",
        description="Optional system prompt prepended to the conversation.",
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        description="Ollama model name to use.",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response (must be False for JSON responses).",
    )


class GenerateResponse(BaseModel):
    """Response body from the /generate endpoint."""

    response: str = Field(..., description="Generated text from the model.")
    model: str = Field(..., description="Model that produced the response.")
    done: bool = Field(..., description="Whether generation is complete.")


@app.get("/health")
async def health() -> dict:
    """
    Health-check endpoint.

    Returns a simple status dict. Used by Docker healthcheck and
    external callers to verify the service is up.
    """
    return {"status": "ok", "service": "llm-spam-detection"}


@app.get("/models")
async def list_models() -> dict:
    """
    List models currently available in the local Ollama instance.

    Proxies the GET /api/tags call to Ollama and returns the raw list.

    Raises:
        HTTPException 502: When Ollama is unreachable.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            return resp.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Ollama unreachable: {exc}") from exc


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Forward a generation request to Ollama and return the model response.

    Constructs an Ollama-compatible payload from the incoming request,
    sends it to the local Ollama API, and returns the model's text output.

    Args:
        request: GenerateRequest containing prompt, optional system prompt, and model name.

    Returns:
        GenerateResponse with the model's text reply, model name, and done flag.

    Raises:
        HTTPException 502: When Ollama is unreachable or returns a non-2xx status.
        HTTPException 500: For unexpected errors during proxying.
    """
    payload = {
        "model": request.model,
        "prompt": request.prompt,
        "stream": False,
    }
    if request.system:
        payload["system"] = request.system

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Ollama returned {exc.response.status_code}: {exc.response.text}",
            ) from exc
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Ollama unreachable: {exc}") from exc

    data = resp.json()
    return GenerateResponse(
        response=data.get("response", ""),
        model=data.get("model", request.model),
        done=data.get("done", True),
    )
