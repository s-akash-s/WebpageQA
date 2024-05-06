from bs4 import BeautifulSoup
import requests
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

def scrape_webpage(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text
    text = soup.get_text()
    
    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    
    # Drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text




# Download the necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    
    return ' '.join(words)

# # Use the function
# url = "https://en.wikipedia.org/wiki/Generative_artificial_intelligence"
# text = scrape_webpage(url)
# preprocessed_text = preprocess_text(text)

# contnet_input_text= preprocessed_text.encode('utf-8')

llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q8_0.bin",
                    model_type="llama",
                    config={'max_new_tokens':300,
                            'temperature':0.01})

def LLamaresponse(input_text, question):
    llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q8_0.bin",
                      model_type="llama",
                      config={'max_new_tokens':300,
                              'temperature':0.01})

    template="""
       Given the context: {input_text}, answer the following question: {question}.
    """
    prompt=PromptTemplate(input_variables=['input_text', 'question'],
                         template=template)

    response= llm(prompt.format(input_text=input_text, question=question))
    print(response)
    return response

def summarize_and_answer(input_text, question):
    # Step 1: Summarization
    # Format the prompt for summarization based on the question
    summarization_prompt = f"Given the context: {input_text}, summarize the main points related to: {question}"
    
    # Use the Llama model to generate the summary
    summary = llm(summarization_prompt)
    
    # Step 2: Question Answering
    # Format the prompt for question answering
    qa_prompt = f"Given the context: {summary}, answer the following question: {question}"
    
    # Use the Llama model to generate the answer
    answer = llm(qa_prompt)
    
    return answer

st.set_page_config(page_title = "Question Answering",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Question Answering 🤖")

url = st.text_input("Enter the URL")
question=st.text_input("Enter the question")

submit=st.button("Generate Answer")

if submit:
    # Scrape and preprocess the webpage after the 'Generate Answer' button is clicked
    text = scrape_webpage(url)
    input_text = preprocess_text(text)
    
    st.write(summarize_and_answer(input_text, question))
