from .model import Model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Llama3(Model):
    """
    Llama 3 model implementation using HuggingFace transformers.
    Supports Llama 3 8B and 70B instruct models.
    """
    def __init__(self, model_size: str = "8B") -> None:
        super().__init__()
        self.model_size = model_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_name = f"meta-llama/Meta-Llama-3-{model_size}-Instruct"
        self.model = self.__load_model()
        self.tokenizer = self.__load_tokenizer()

    def get_context_length(self) -> int:
        # Llama 3 supports 8k context window
        return 8192

    def encode_text(self, text: str) -> list:
        """
        This method encodes the text

        :param text: text to encode
        :return: encoded text
        """
        return self.tokenizer.encode(text)

    def __load_model(self):
        """Load the Llama 3 model"""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        return model

    def __load_tokenizer(self):
        """Load the tokenizer for Llama 3"""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def generate_output(self, input: str, max_new_tokens: int, temperature: float = 0.7) -> str:
        """
        This method generates the output given the input

        :param input: input to the model
        :param max_new_tokens: maximum number of tokens to generate
        :param temperature: temperature parameter for the model
        :return: output of the model
        """
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant for conducting meta-analyses of randomized controlled trials."},
                {"role": "user", "content": input},
            ]

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode only the new tokens
            response = self.tokenizer.decode(
                outputs[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )
            return response

        except Exception as e:
            print(f"[ERROR] Llama 3 generation failed: {e}")
            return "Error: Llama 3 generation failed."


class Llama38B(Llama3):
    """Llama 3 8B Instruct model"""
    def __init__(self) -> None:
        super().__init__(model_size="8B")


class Llama370B(Llama3):
    """Llama 3 70B Instruct model"""
    def __init__(self) -> None:
        super().__init__(model_size="70B")
