from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

# ======================
# Structured Output Model
# ======================

class Person(BaseModel):
    name: str
    age: int
    profession: str


parser = PydanticOutputParser(pydantic_object=Person)

# ======================
# Prompt Template
# ======================

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert information extractor."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{text}"),
    ("system", "Extract the information.\n{format_instructions}")
]).partial(format_instructions=parser.get_format_instructions())


# ======================
# LLM
# ======================

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

chain = prompt | llm | parser

# ======================
# Chat History
# ======================

chat_history = []

while True:
    user_input = input("\nEnter text: ")

    if user_input.lower() == "exit":
        break

    result = chain.invoke({
        "text": user_input,
        "chat_history": chat_history
    })

    chat_history.append(("human", user_input))
    chat_history.append(("ai", str(result)))

    print("\nExtracted Data:")
    print(result)