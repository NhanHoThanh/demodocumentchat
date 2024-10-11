from docx import Document
import openai
import requests
from supabase import create_client, Client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import SupabaseVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize Supabase Client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(supabase_url, supabase_key)

# Step 2: Load and Split the Document


def load_and_split_document(file_path):
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            document = file.read()
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        document = '\n'.join([para.text for para in doc.paragraphs])
    else:
        raise ValueError(
            "Unsupported file format. Only .txt and .docx are supported.")

    splitter = RecursiveCharacterTextSplitter(max_length=1000, overlap=100)
    chunks = splitter.split_text(document)
    return chunks

# Step 3: Embed the Text


def embed_text(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    embedded_chunks = [embeddings.embed(chunk) for chunk in chunks]
    return embedded_chunks

# Step 4: Store Vectors in Supabase


def store_vectors_in_supabase(embedded_chunks):
    for i, vector in enumerate(embedded_chunks):
        supabase.table('document').insert(
            {'id': i, 'vector': vector}).execute()

# Step 5: Query Processing


def process_query(question):
    # Embed the question
    embeddings = OpenAIEmbeddings()
    question_vector = embeddings.embed(question)

    # Query Supabase for relevant chunks
    response = supabase.rpc(
        'match_vectors', {'query_vector': question_vector}).execute()
    relevant_chunks = response.data

    # Generate response using OpenAI's chat API
    context = ' '.join([chunk['text'] for chunk in relevant_chunks])
    chat_response = openai.Completion.create(
        engine="davinci",
        prompt=f"Context: {context}\n\nQuestion: {question}\nAnswer:",
        max_tokens=150
    )
    return chat_response.choices[0].text.strip()

# Main Function


def main():
    file_path = 'path_to_your_document.txt'
    question = 'your_question_here'

    # Load and split the document
    chunks = load_and_split_document(file_path)

    # Embed the text
    embedded_chunks = embed_text(chunks)

    # Store vectors in Supabase
    store_vectors_in_supabase(embedded_chunks)

    # Process the query
    answer = process_query(question)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
