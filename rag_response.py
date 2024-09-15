from langchain_core.messages import HumanMessage
from langchain_cohere import ChatCohere
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import requests
from bs4 import BeautifulSoup
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

# Singleton pattern to manage the store variable
class StoreManager:
    _instance = None
    store = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StoreManager, cls).__new__(cls)
        return cls._instance

    def get_store(self):
        return self.store

def extract_artifact_details(soup):
    artifact_div = soup.find('div', id='artifact-details')
    paragraphs = artifact_div.find_all('p')
    details_data = []
    for p in paragraphs:
        details_data.append(" ".join(p.get_text().strip().split()))
    artifact_details = "\n".join(details_data)
    return artifact_details

def extract_artifact_categories(soup):
    categories = soup.find_all('div', class_="category")
    additional_data = []
    for category in categories:
        category_text = " ".join(category.find_parent('div').get_text().strip().split())
        additional_data.append(category_text)
    artifact_categories = "\n".join(additional_data)
    return artifact_categories

def extract_data_from_artifact(artifact_id):
    artifact_url = "https://antiquities.bibalex.org/Collection/Detail.aspx?lang=en&a={}".format(artifact_id)
    response = requests.get(artifact_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    artifact_categories = extract_artifact_categories(soup)
    artifact_details = extract_artifact_details(soup)
    artifact_info = {
        "categories": artifact_categories,
        "details": artifact_details,
    }
    return artifact_info

def load_template():
    template = """Using the following pieces of context about an artifact's detailed information
        and description answer the following question as if you are a tour guide. If you don't know the answer
        to the question, say that you don't know. Keep your answer brief and answer the question without introductions.

        Artifact Information:
        {categories}

        Artifact Details:
        {details}

        Question: {question}

        Guided Answer:"""
    return template

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    store_manager = StoreManager()
    store = store_manager.get_store()
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    print(store)
    return store[session_id]

def initialize_model(template):
    llm = ChatCohere(model="command-r", cohere_api_key = "6uXCCMmaRpobd1zpyxC8OaOJ719NhqxmPXzAenue")
    custom_rag_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"),
        (template),
    ])
    runnable = custom_rag_prompt | llm
    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    return with_message_history

if __name__ == '__main__':
    store = {}
    while True:
        artifact_id = input("Enter artifact id:") # should be an input
        question = input("what is your question? ")
        artifact_info = extract_data_from_artifact(artifact_id)
        template = load_template()  
        with_message_history = initialize_model(template)
        response = with_message_history.invoke(
            {"categories": artifact_info["categories"], "details": artifact_info["details"], "question": question},
            config={"configurable": {"session_id": str(artifact_id)}},
        ).content
        print(response)