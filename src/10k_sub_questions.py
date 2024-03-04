import json
import os
import time

from dotenv import find_dotenv, load_dotenv
import nest_asyncio
import tiktoken
import streamlit as st
from llama_index.core import (GPTVectorStoreIndex, Settings, SimpleDirectoryReader, 
                              ServiceContext)
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from pyprojroot import here

from web_retrieval import evaluate_response_usefulness, search_web


# Apply asyncio patch
nest_asyncio.apply()

# Load environment variables
_ = load_dotenv(find_dotenv())

class DocumentLoader:
    @staticmethod
    def load_documents(documents_dir):
        documents_path = here(documents_dir)
        documents = SimpleDirectoryReader(
            input_files=[os.path.join(documents_path, d) for d in os.listdir(documents_path)]
        ).load_data()
        return documents

class QueryEngineManager:
    def __init__(self, documents_paths,company):
        self.documents_paths = documents_paths
        self.query_engines = []
        self.company=company
    
    def load_engines(self, llm, embed_model):
        for year, path in self.documents_paths.items():
            docs = DocumentLoader.load_documents(path)
            index = GPTVectorStoreIndex.from_documents(docs)
            engine = index.as_query_engine(similarity_top_k=3)
            query_engine_tool=QueryEngineTool(
                            query_engine=engine,
                            metadata= ToolMetadata(name=f"{self.company}_10k_{year}",description=f"Provide information about {self.company} financials for year 2019"))
            self.query_engines.append(query_engine_tool)
        return self.query_engines

# Assuming the setup for tokenizer, Settings, etc., remains as in your original script
def setup_token_counter():
    return TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    )

def load_questions_config(path_to_config, company_name):
    with open(path_to_config, "r") as file:
        questions_config = json.load(file)
        # Replace the placeholder {company} with the actual company name
        for item in questions_config:
            item["question"] = item["question"].replace("{company}", company_name)
    return questions_config

def process_questions(search_engine,questions_config,company_name):
    final_answers = ""
    for item in questions_config:
        sector = item["sector"]
        question = item["question"]
        response = search_engine.query(question)
        
        is_useful = evaluate_response_usefulness(question, response)
        print(f"{sector}: ", is_useful)
        
        if is_useful == "No":
            response = search_web(company_name, question)
        
        final_answers += f"{sector}:\n{response}\n"
        # Add delay or further processing as needed
        time.sleep(5)
    
    return final_answers

# Main workflow
def main():
    st.title("Company Document Analysis with LLaMA")

    # Input fields for company name and data path
    company = st.text_input("Company Name", value="Nvidia")
    data_path = st.text_input("Data Path", value="data/Nvidia/docs_")

    # Button to trigger analysis
    if st.button("Analyze Documents"):
        # Setup and process documents
        final_answers = analyze_documents(company, data_path)
        st.text_area("Analysis Results", value=final_answers, height=300)

def analyze_documents(company, data_path):
    llm = OpenAI(temperature=0.0, model="gpt-3.5-turbo")
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    token_counter = setup_token_counter()
    Settings.callback_manager = CallbackManager([token_counter])

    years = ["2019", "2020", "2021", "2022", "2023"]
    documents_paths = {year: f"{data_path}{year}" for year in years}

    engine_manager = QueryEngineManager(documents_paths, company)
    query_engines_tools = engine_manager.load_engines(llm, embed_model)

    s_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engines_tools, use_async=True, verbose=False)
    questions_config_path = "src/configs/questions.json"
    questions_config = load_questions_config(questions_config_path, company)
    final_answers = process_questions(s_engine,questions_config, company)
    return final_answers

if __name__ == "__main__":
    main()