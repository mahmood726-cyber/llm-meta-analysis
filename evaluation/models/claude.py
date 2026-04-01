from .model import Model
from anthropic import Anthropic
import time

REQ_TIME_GAP = 5
MAX_API_RETRY = 3

class Claude(Model):
    """
    Claude model implementation using Anthropic API.
    Supports Claude 3 Opus, Sonnet, and Haiku models.
    """
    def __init__(self, model_version: str = "claude-3-5-sonnet-20241022") -> None:
        super().__init__()
        self.model_version = model_version
        self.client = Anthropic()
        self.context_lengths = {
            "claude-3-5-sonnet-20241022": 200000,
            "claude-3-5-sonnet-20240620": 200000,
            "claude-3-opus-20240229": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-haiku-20240307": 200000,
        }

    def get_context_length(self) -> int:
        return self.context_lengths.get(self.model_version, 200000)

    def encode_text(self, text: str) -> list:
        """
        This method encodes the text using Anthropic's tokenizer.
        Note: Anthropic uses a different tokenization approach.

        :param text: text to encode
        :return: Approximate token count (using a simple estimation)
        """
        # Anthropic doesn't expose a direct tokenizer, so we use a rough estimation
        # Approximately 4 characters per token
        return len(text) // 4

    def generate_output(self, input: str, max_new_tokens: int, temperature: float = 1.0) -> str:
        """
        This method generates the output given the input

        :param input: input to the model
        :param max_new_tokens: maximum number of tokens to generate
        :param temperature: temperature parameter for the model
        :return: output of the model
        """
        completion = None
        for attempt in range(MAX_API_RETRY):
            try:
                completion = self.client.messages.create(
                    model=self.model_version,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    system="You are a helpful assistant for conducting meta-analyses of randomized controlled trials.",
                    messages=[
                        {"role": "user", "content": input}
                    ]
                )
                break
            except Exception as e:
                print(f"[ERROR] Attempt {attempt + 1}/{MAX_API_RETRY}: {e}")
                if attempt < MAX_API_RETRY - 1:
                    time.sleep(REQ_TIME_GAP)

        if completion is None:
            return "Error: Claude API call failed."

        return completion.content[0].text


class ClaudeOpus(Claude):
    """Claude 3 Opus - Most capable model for complex tasks."""
    def __init__(self) -> None:
        super().__init__(model_version="claude-3-opus-20240229")


class ClaudeSonnet(Claude):
    """Claude 3.5 Sonnet - Balanced performance and speed."""
    def __init__(self) -> None:
        super().__init__(model_version="claude-3-5-sonnet-20241022")


class ClaudeHaiku(Claude):
    """Claude 3 Haiku - Fastest and most cost-effective."""
    def __init__(self) -> None:
        super().__init__(model_version="claude-3-haiku-20240307")
