import openai
import os
# from openai import OpenAI
from time import sleep
# from llama_index import download_loader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import SimpleDirectoryReader, ServiceContext, GPTVectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.readers.web import SimpleWebPageReader
from bs4 import BeautifulSoup

from googleapiclient.discovery import build
import pprint
import requests
from tqdm import tqdm
from typing import List
import instructor
from pydantic import BaseModel, field_validator, ValidationError

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
search_api_key = "search_api_key" 
search_cse_id = "search_cse_id" 

#This enables response_model keyword in openai.ChatCompletion
client = instructor.patch(openai.OpenAI())

class SearchAPIResponse(BaseModel):
    output:str
    @field_validator('output')
    def output_must_be_yes_or_no(cls, v):
        if v.lower() not in ["yes", "no"]:
            raise ValueError('output must be "yes" or "no"')
        return v.lower()

class KeywordAPIResponse(BaseModel):
    keywords: List[str]

def extract_keywords(question):
    task_prompt=f"""Extract at most three keywords separated by comma from the following dialogues and questions as queries for the web search, including topic background within dialogues and main intent within questions.
    question: What is Henry Feilden’s occupation? 
    query: [Henry Feilden, occupation] 
    question: In what city was Billy Carlson born? 
    query: [city, Billy Carlson, born] 
    question: What is the religion of John Gwynn? 
    query: [religion, John Gwynn] 
    question: What sport does Kiribati men’s national basketball team play? 
    query: [sport, Kiribati men’s national basketball team play]
    question: {{question}}
    query:"""

    # client = openai.OpenAI()
    prompt_template = task_prompt
    inputs = prompt_template.format(question=question)
    messages = [{"role": "user", "content": inputs}]
    try:
        completion= client.chat.completions.create(model="gpt-3.5-turbo", temperature=0.1,messages=messages,response_model=KeywordAPIResponse)
        print("completion: ",completion.keywords,type(completion.keywords))
        # results = completion.choices[0].message.content #completion["choices"][0]["message"]["content"]
        results=completion.keywords
        return results
    except Exception as e:
        print('Rate limit error: ',e)        
        
def evaluate_response_usefulness(question,response):
    task_prompt=f"""I asked the following question to my language model: {{question}}
    and this is the response I got: {{response}}
    Do you think this response is useful and provides all the information asked in the question? Was all the information mentioned/present in the context/documents? (Answer in Yes or No only)"""
    # openai.api_key = openai_key
    queries = []
    # client = openai.OpenAI()
    inputs = task_prompt.format(question=question,response=response)
    messages = [{"role": "user", "content": inputs}]
    print(messages)
    try:
        completion:SearchAPIResponse = client.chat.completions.create(model="gpt-3.5-turbo", temperature=0.1,messages=messages,response_model=SearchAPIResponse)
        result=completion.output
        print("result: ",result)
        # result = completion.choices[0].message.content #completion["choices"][0]["message"]["content"]
        return result
    except Exception as e:
        print('Exception: ',e)

def scrape(sites):
    urls=[]
    def scrape_helper(current_site):
        nonlocal urls
        r=requests.get(current_site)
        s=BeautifulSoup(r.text,"html.parser")
        for i in s.find_all("a"):
            if "href" in i.attrs:
                href=i.attrs["href"]
                if href.startswith("/") or href.startswith("#"):
                    full_url=site+href
                    if full_url not in urls:
                        urls.append(full_url)
                        scrape_helper(full_url)
    for site in sites:
        scrape_helper(site)
    return urls

def load_docs_to_gpt_vector_store(urls):
    # urls=scrape(sites)
    print("URLs: ",urls)
    # bsWebReader=download_loader("BeautifulSoupWebReader")
    loader=SimpleWebPageReader(html_to_text=True) #bsWebReader()
    documents=loader.load_data(urls)
    parser=SimpleNodeParser()

    nodes=parser.get_nodes_from_documents(documents)
    llm=OpenAI(temperature=0.0, model="gpt-3.5-turbo")
    # llm_predictor=LLMPredictor(llm=llm)
    embed_model=OpenAIEmbedding(model="text-embedding-ada-002",)
    # service_context=ServiceContext.from_defaults(llm_predictor=llm_predictor)
    max_input_size=4096
    num_output=256
    max_chunk_overlap=20
    chunk_overlap_ratio=0.2
    prompt_helper=PromptHelper(max_input_size,num_output,chunk_overlap_ratio)

    service_context=ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
            prompt_helper=prompt_helper 
        )
    index=GPTVectorStoreIndex(nodes,service_context=service_context)
    index.storage_context.persist("src/indexes")
    return index

def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res

def chat(query,index):
    storage_context = StorageContext.from_defaults(persist_dir="src/indexes")
    index=load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    response=query_engine.query(query)
    return response

def search_web(company,question):
    keywords=extract_keywords([question])
    formatted_keywords=[]
    for keyword in keywords:
        print(keyword,company)
        if company not in keyword:
            formatted_keywords.append(company+" "+keyword)
        else:
            formatted_keywords.append(keyword)
    print("formatted keywords: ",formatted_keywords)
    urls=set()
    for keyword in formatted_keywords:
        results = google_search(keyword, search_api_key, search_cse_id, num=3,) #dateRestrict='m3'
        for result in results["items"]:
            urls.add(result['link'])
    idx=load_docs_to_gpt_vector_store(list(urls))
    response=chat(question,idx)
    return response
