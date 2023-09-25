import sys
import time
import traceback
from datetime import datetime
from typing import List, Union, Optional

import openai
import timeout_decorator
from dotenv import dotenv_values
from inputimeout import inputimeout, TimeoutOccurred
from typing_extensions import TypedDict

from langprop.prompt import parse_template

config = dotenv_values(".env")


class Role:
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


def init_openai():
    global api_initialized
    if not api_initialized:
        openai.api_type = config["OPENAI_API_TYPE"]
        openai.api_base = config["OPENAI_API_BASE"]
        openai.api_version = config["OPENAI_API_VERSION"] if config["OPENAI_API_TYPE"] == "azure" else ""
        openai.api_key = config["OPENAI_API_KEY"]
        api_initialized = True


api_initialized = False
init_openai()


class LMQuery(TypedDict):
    role: str
    content: str


@timeout_decorator.timeout(600, use_signals=False)
def call_chatgpt(messages: List[LMQuery], n=1):
    kwargs = {"engine": config["OPENAI_API_ENGINE"]} if config["OPENAI_API_TYPE"] == "azure" else {"model": config["OPENAI_MODEL"]}
    result = openai.ChatCompletion.create(
        messages=messages,
        n=n,
        timeout=300,
        **kwargs
    )
    return [elem["message"]["content"] for elem in result["choices"]]


def prompt_to_query(prompt, init_role=Role.USER) -> List[LMQuery]:
    results = []
    query = {"role": init_role, "content": ""}
    for line in prompt.split("\n"):
        if line[:3] == "%% ":
            if query["content"]:
                results.append(query)
            role = line[3:].strip("\n").strip(" ").lower()
            query = {"role": role, "content": ""}
        else:
            query["content"] += line + "\n"
    if query["content"]:
        results.append(query)
    return results


def template_to_query(template, **local_dict):
    prompt = parse_template(template, **local_dict)
    query = prompt_to_query(prompt)
    return prompt, query


class LangAPI:
    def __init__(self, n_responses: int = 1, n_tries: int = 10):
        self.n_responses = n_responses
        self.n_tries = n_tries
        self.wait_api_call = 0

    def call_llm(self, query):
        return call_chatgpt(query, n=self.n_responses)

    def __call__(self, query: Union[List[LMQuery], str], n_tries: Optional[int] = None) -> List[str]:
        if n_tries is None:
            n_tries = self.n_tries
        if isinstance(query, str):
            query = prompt_to_query(query)
        if self.wait_api_call > 0:
            time.sleep(self.wait_api_call)
        try:
            response = self.call_llm(query)
            self.wait_api_call *= 0.5
            if self.wait_api_call < 0.05:
                self.wait_api_call = 0
            return response
        except (timeout_decorator.TimeoutError, openai.error.RateLimitError, Exception) as e:
            print(type(e))
            if n_tries > 0:
                print(f"API request failed with exception {e}. Number of tries remaining: {n_tries}. Current time: {datetime.now()}. Retrying in {self.wait_api_call} seconds...")
                self.wait_api_call += 2
                return self(query, int(n_tries) - 1)
            traceback.print_exc()
            print("Your prompt was as follows: ")
            for q in query:
                print(f"{q['role']}: ")
                print(q["content"])

            try:
                retry = inputimeout(prompt="Do you want to retry? (y/n): ", timeout=120)
                while True:
                    if retry == "y":
                        return self(query)
                    elif retry == "n":
                        print("Exiting...")
                        sys.exit(-1)
                    else:
                        retry = input("Sorry, we do not recognise your input. Do you want to retry? (y/n): ").lower()
            except TimeoutOccurred:
                return self(query)


if __name__ == "__main__":
    lang_api = LangAPI()
    response = lang_api("hello!")
    print(response)
