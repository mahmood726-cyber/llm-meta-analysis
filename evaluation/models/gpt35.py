from .model import Model
import time
try:
    from openai import OpenAI
except ImportError:  # optional provider SDK for offline tests
    OpenAI = None
try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency in tests
    tiktoken = None

REQ_TIME_GAP = 5 #
MAX_API_RETRY = 3

class GPT35(Model):
    def __init__(self) -> None:
        super().__init__()
        try:
            self.client = OpenAI() if OpenAI is not None else None
        except Exception:
            self.client = None
        self.encoder = tiktoken.get_encoding("cl100k_base") if tiktoken is not None else None

    def get_context_length(self) -> int:
        return 16385
    
    def encode_text(self, text: str) -> str:
        """
        This method encodes the text

        :param text: text to encode

        :return encoded text
        """
        if self.encoder is None:
            return list(text.encode("utf-8"))
        return self.encoder.encode(text)
    
    def generate_output(self, input: str, max_new_tokens: int, temperature: str = 1) -> str:
        """
        This method generates the output given the input

        :param input: input to the model
        :param max_new_tokens: maximum number of tokens to generate
        :param temperature: temperature parameter for the model

        :return output of the model
        """
        if self.client is None:
            return "Error: GPT-3.5 client is unavailable."

        completion = None
        for _ in range(MAX_API_RETRY):
            try:
                completion = self.client.chat.completions.create(
                    model="gpt-3.5-turbo-0125", #16k tokens (https://platform.openai.com/docs/models/gpt-3-5-turbo)
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for conducting meta-analyses of randomized controlled trials."},
                        {"role": "user", "content": input}
                    ],
                    # TODO: currently set as default but should figure out temperature/top_p parameters
                    # https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api/172683
                    temperature=temperature,
                    top_p=1,
                    max_tokens=max_new_tokens,
                )
            except Exception as e:
                print("[ERROR]", e)
                time.sleep(REQ_TIME_GAP)
                
        if completion is None:
            return "Error: GPT-3.5 API call failed."
        else:
            return completion.choices[0].message.content
