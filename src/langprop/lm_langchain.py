from dotenv import dotenv_values

from langprop.lm_api import init_openai
from langchain.llms import OpenAI
config = dotenv_values(".env")

init_openai()

llm = OpenAI(temperature=0.9, engine=config["OPENAI_API_ENGINE"])

text = "Explain the concept of machine learning in one paragraph"
print(llm(text))
