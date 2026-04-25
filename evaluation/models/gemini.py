from .model import Model
import time
import os

# Lazy import: google.generativeai pulls a 12s import-tree (gRPC + protobuf
# + auth) which Overmind's smoke importer flagged as a 300s cumulative
# timeout across all eval/models. Deferring to instance __init__ keeps
# the module import-light (lessons.md heavy-ML-imports rule, 2026-04-25).

REQ_TIME_GAP = 5
MAX_API_RETRY = 3

class Gemini(Model):
    """
    Gemini model implementation using Google Generative AI API.
    Supports Gemini Pro and Gemini Ultra models.
    """
    def __init__(self, model_version: str = "gemini-1.5-pro") -> None:
        super().__init__()
        import google.generativeai as genai  # lazy: see module docstring
        self._genai = genai
        self.model_version = model_version
        # Configure the API - expects GOOGLE_API_KEY environment variable
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable must be set")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_version)
        self.context_lengths = {
            "gemini-1.5-pro": 1000000,
            "gemini-1.5-flash": 1000000,
            "gemini-pro": 91728,
        }

    def get_context_length(self) -> int:
        return self.context_lengths.get(self.model_version, 1000000)

    def encode_text(self, text: str) -> list:
        """
        This method encodes the text using Gemini's tokenizer.

        :param text: text to encode
        :return: Approximate token count
        """
        # Gemini uses a similar tokenization approach to other models
        # Using a rough estimation of 4 characters per token
        return len(text) // 4

    def generate_output(self, input: str, max_new_tokens: int, temperature: float = 1.0) -> str:
        """
        This method generates the output given the input

        :param input: input to the model
        :param max_new_tokens: maximum number of tokens to generate
        :param temperature: temperature parameter for the model
        :return: output of the model
        """
        generation_config = self._genai.types.GenerationConfig(
            max_output_tokens=max_new_tokens,
            temperature=temperature,
        )

        for attempt in range(MAX_API_RETRY):
            try:
                response = self.client.generate_content(
                    input,
                    generation_config=generation_config,
                    safety_settings={
                        'HATE': 'BLOCK_NONE',
                        'HARASSMENT': 'BLOCK_NONE',
                        'SEXUAL': 'BLOCK_NONE',
                        'DANGEROUS': 'BLOCK_NONE',
                    }
                )
                return response.text
            except Exception as e:
                print(f"[ERROR] Attempt {attempt + 1}/{MAX_API_RETRY}: {e}")
                if attempt < MAX_API_RETRY - 1:
                    time.sleep(REQ_TIME_GAP)

        return "Error: Gemini API call failed."


class GeminiPro(Gemini):
    """Gemini 1.5 Pro - High capability model."""
    def __init__(self) -> None:
        super().__init__(model_version="gemini-1.5-pro")


class GeminiFlash(Gemini):
    """Gemini 1.5 Flash - Fast and cost-effective."""
    def __init__(self) -> None:
        super().__init__(model_version="gemini-1.5-flash")
