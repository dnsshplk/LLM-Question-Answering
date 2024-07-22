import os
from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_chroma import Chroma

from langchain_community.document_loaders import PyPDFLoader


def load_document(file_name, folder_name = 'data'):
    data_path = os.path.join(folder_name, file_name)
    loader = PyPDFLoader(data_path)
    document = loader.load()
    return document


def split_text(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100,
        add_start_index = True
    )
    chunks = text_splitter.split_documents(document)
    return chunks


def create_db(chunks, embedding_function, persist_directory = './chroma_db'):
    db = Chroma.from_documents(chunks, embedding_function, persist_directory=persist_directory)
    print(f'Saved {len(chunks)} chunks into {persist_directory}')


def main():
    document = load_document('Robinson_Crusoe.pdf')

    chunks = split_text(document)

    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    create_db(chunks, embedding_function)


if __name__ == '__main__':
    main()