import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("📄 Document question answering")
import streamlit.components.v1 as components
from PyPDF2 import PdfReader
from langchain import LLMChain, PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from googletrans import Translator
from langdetect import detect
from pdf2docx import Converter
from io import BytesIO
import tempfile
import os

# Set up Streamlit page configuration
st.set_page_config(page_title="PDFOasis", page_icon="📚", layout="wide")

# Hide Streamlit's default hamburger menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Set up the main title and description
st.title("📚 PDFOasis")
st.markdown("**An Intelligent PDF Query and Analysis System**")

# Initialize necessary components
def initialize_components():
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key="AIzaSyBMSi3Bx9WqfQPHACdfdCCcGsSHBRvUieI"
    )

    # Initialize Translator
    translator = Translator()

    return llm, translator

llm, translator = initialize_components()

# Define the Summarization Chain
def create_summarization_chain(llm):
    summarization_prompt = """
    You are a summarization assistant. Summarize the following text comprehensively while retaining all the key points:

    {text}

    Summary:
    """
    prompt = PromptTemplate(template=summarization_prompt, input_variables=["text"])
    return LLMChain(llm=llm, prompt=prompt)

# Define the QA Chain
def create_qa_chain(llm):
    return load_qa_chain(llm, chain_type="stuff")

# Function to process uploaded PDF and create necessary data structures
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(file.read())
        pdf_reader = PdfReader(temp_pdf.name)
        raw_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text

    # Detect language
    pdf_language = detect(raw_text)

    # Split text into manageable chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    
    # Generate embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key="AIzaSyBMSi3Bx9WqfQPHACdfdCCcGsSHBRvUieI"
    )

    # Create vector store for similarity search
    docsearch = FAISS.from_texts(texts, embeddings)

    return docsearch, pdf_language, texts, raw_text

# Function to convert PDF to DOCX
def convert_pdf_to_docx(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_bytes)
        temp_pdf_path = temp_pdf.name

    docx_file = BytesIO()
    cv = Converter(temp_pdf_path)
    cv.convert(docx_file)
    cv.close()
    os.unlink(temp_pdf_path)
    return docx_file.getvalue()

# Function to convert PDF to TXT
def convert_pdf_to_txt(raw_text):
    return raw_text

# Centered file uploader
def file_uploader():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader("Upload your PDF file here", type=["pdf"])
    return uploaded_file

# Define the Agent
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate

class PDFOasisAgent:
    def __init__(self, summarization_chain, qa_chain, docsearch):
        self.summarization_chain = summarization_chain
        self.qa_chain = qa_chain
        self.docsearch = docsearch

    def run(self, action, text):
        if action == "summarization":
            return self.summarization_chain.run(text=text).strip()
        elif action == "qa":
            docs = self.docsearch.similarity_search(text)
            return self.qa_chain.run(input_documents=docs, question=text).strip()
        return None

# Function to translate text with error handling
def translate_text(text, target_language):
    try:
        translated = translator.translate(text, dest=target_language)
        return translated.text
    except Exception as e:
        return f"Error translating text: {str(e)}"



# Main application logic
def main():
    uploaded_file = file_uploader()

    if uploaded_file is not None:
        with st.spinner("Processing your PDF..."):
            docsearch, pdf_language, texts, raw_text = process_pdf(uploaded_file)
        st.success(f"PDF processed successfully! Detected language: **{pdf_language.upper()}**")

        # Initialize chains and agent
        summarization_chain = create_summarization_chain(llm)
        qa_chain = create_qa_chain(llm)
        agent = PDFOasisAgent(summarization_chain, qa_chain, docsearch)

        # Sidebar for additional features
        with st.sidebar:
            st.header("📚 PDFOasis")

            # Summarization Section
            st.subheader("🔍 Summarization")
            summary_option = st.selectbox("Choose summarization type:", ["Full Text Summary", "Section Summary"])
            if summary_option == "Full Text Summary":
                if st.button("Generate Full Summary"):
                    with st.spinner("Generating summary..."):
                        summary = agent.run("summarization", raw_text)
                    st.success("Summary generated successfully!")
                    st.text_area("Summary:", value=summary, height=200)
                    
                    
            else:
                start_page = st.number_input("Start Page:", min_value=1, max_value=len(texts), value=1)
                end_page = st.number_input("End Page:", min_value=1, max_value=len(texts), value=1)
                if st.button("Generate Section Summary"):
                    if start_page <= end_page:
                        with st.spinner("Generating summary..."):
                            section_text = ' '.join(texts[start_page-1:end_page])
                            summary = agent.run("summarization", section_text)
                        st.success("Summary generated successfully!")
                        st.text_area("Section Summary:", value=summary, height=200)
                        
                        
                    else:
                        st.error("Start page should be less than or equal to end page.")

            st.markdown("---")

            # Query Section
            st.subheader("❓ Query")
            query_input = st.text_input("Enter your query:")
            if st.button("Ask Query"):
                if query_input:
                    with st.spinner("Searching for answers..."):
                        answer = agent.run("qa", query_input)
                        st.success("Answer found!")
                        st.text_area("Answer:", value=answer, height=200)
                        
                        
                else:
                    st.error("Please enter a query.")

            st.markdown("---")

           

            # Translate Section
            st.subheader("🌍 Translate Answer")
            if st.text_area("Answer:", key="answerInput"):
                target_language = st.selectbox("Select target language for translation:", ["en", "es", "fr", "de", "zh"])  # Add more languages as needed
                if st.button("Translate Answer"):
                    translated_answer = translate_text(st.session_state.answerInput, target_language)
                    st.text_area("Translated Answer:", value=translated_answer, height=200)

            st.markdown("---")

            # Download Section
            st.subheader("📥 Download PDF in Other Formats")
            download_format = st.selectbox("Choose the format to download:", ["DOCX", "TXT"])
            if st.button("Download PDF"):
                with st.spinner("Converting PDF..."):
                    if download_format == "DOCX":
                        docx_data = convert_pdf_to_docx(uploaded_file.read())
                        st.success("PDF converted to DOCX successfully!")
                        st.download_button(label="Download DOCX", data=docx_data, file_name="converted.docx")
                    elif download_format == "TXT":
                        txt_data = convert_pdf_to_txt(raw_text)
                        st.success("PDF converted to TXT successfully!")
                        st.download_button(label="Download TXT", data=txt_data, file_name="converted.txt")

if __name__ == "__main__":
    main()

st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:

        # Process the uploaded file and question.
        document = uploaded_file.read().decode()
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question}",
            }
        ]

        # Generate an answer using the OpenAI API.
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True,
        )

        # Stream the response to the app using `st.write_stream`.
        st.write_stream(stream)
