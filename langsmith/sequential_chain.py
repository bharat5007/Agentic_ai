import truststore
truststore.inject_into_ssl()

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "Sequential LLM APP"

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
model2 = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.8)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
    'run_name': 'report_generation_chain',   # this will override the default name we set in line 12 and will be used in the UI to identify this run
    'tags': ['llm app', 'report generation'],
    'metadata': {
        'author': 'John Doe',
        'description': 'A chain that generates a report and then summarizes it.'
    }
}
result = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result)