import base64
import time
import os

os.environ["HTTP_PROXY"] = "http://localhost:1080"
os.environ["HTTPS_PROXY"] = "http://localhost:1080"

import subprocess
from typing import Any
from datetime import datetime, timedelta

import openai
from openai import OpenAI

from eval_types import MessageList, SamplerBase

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """
    
    # Class-level variables for sharing tokens across instances
    api_key = None
    token_expiry = None

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        base_url: str | None = None,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        # Initialize class-level api_key if not already set
        if ChatCompletionSampler.api_key is None:
            ChatCompletionSampler.api_key = os.environ.get(self.api_key_name)
        self.base_url = base_url
        self._refresh_token_if_needed(self.base_url)
        self.client = OpenAI(api_key=ChatCompletionSampler.api_key, base_url=self.base_url)
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    @classmethod
    def _refresh_token_if_needed(cls, base_url=None):
        """Refresh the Google Cloud token if it's expired or about to expire (within 5 minutes)"""
        if not cls.api_key and base_url:
            current_time = datetime.now()
            if not cls.token_expiry or current_time + timedelta(minutes=5) >= cls.token_expiry:
                try:
                    print("Fetching new token from gcloud.")
                    result = subprocess.run(
                        "gcloud auth print-access-token",
                        capture_output=True,
                        text=True,
                        check=True,
                        shell=True
                    )
                    cls.api_key = result.stdout.strip()
                    # Set token expiry to 55 minutes from now (giving 5-minute buffer)
                    cls.token_expiry = current_time + timedelta(minutes=55)
                except FileNotFoundError:
                    print("gcloud command not found. Please ensure gcloud SDK is installed and in your PATH.")
                    cls.api_key = None
                except subprocess.CalledProcessError as e:
                    print(f"Error fetching token from gcloud: {e}")
                    cls.api_key = None
        elif not cls.api_key:
            cls.api_key = ""

    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                # Refresh token if needed before making the API call
                self._refresh_token_if_needed(self.base_url)
                # Update client with potentially new token
                self.client = OpenAI(api_key=ChatCompletionSampler.api_key, base_url=self.base_url)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                print("Error", e)
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
