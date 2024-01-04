import os
import pdfplumber
import subprocess
import json
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def generate_embedding(text_chunk):
    """
    Generates an embedding for a given text chunk using the 'gte-small' model.

    :param text_chunk: A string representing a text chunk.
    :return: A tensor representing the embedding of the text chunk.
    """
    tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
    model = AutoModel.from_pretrained("thenlper/gte-small")

    # Tokenize the text chunk
    inputs = tokenizer(text_chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Generate model outputs
    outputs = model(**inputs)

    # Pool the embeddings and normalize
    embedding = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
    embedding = F.normalize(embedding, p=2, dim=1)  # Normalizing

    return embedding.squeeze()

def search_most_similar_embedding(prompt, df):
    """
    Search for the chunk index with the highest cosine similarity to the prompt.

    :param prompt: The prompt string to compare against
    :param df: DataFrame containing the text chunks and their embeddings
    :return: Index of the chunk with the highest similarity
    """
    # Generate embedding for the prompt
    prompt_embedding = generate_embedding(prompt)

    max_similarity = -1
    most_similar_index = -1

    for index, row in df.iterrows():
        chunk_embedding = torch.tensor(row['Vector_Embedding'], dtype=torch.float)
        chunk_val = row['Text_Chunk']
        similarity = torch.nn.functional.cosine_similarity(prompt_embedding, chunk_embedding, dim=0)
        #print(f"Chunk Text: {chunk_val}, Similarity: {similarity}")

        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_index = row['Chunk_Index']

    return most_similar_index

def ocr_and_chunk(pdf_path, output_path='ocr_output.pdf'):
    """
    Applies OCR to a PDF file, makes it searchable, extracts text, and splits it into 1000-character chunks.

    :param pdf_path: Path to the source PDF file
    :param output_path: Path for the OCR processed output PDF file
    :return: A list of text chunks, each having 1000 characters
    """
    # Apply OCR to the PDF
    subprocess.run(['ocrmypdf', '--force-ocr', pdf_path, output_path], check=True)

    # Now, extract text from the OCR-processed PDF
    text_chunks = []
    with pdfplumber.open(output_path) as pdf:
        full_text = ''
        for page in pdf.pages:
            full_text += page.extract_text() or ''

        # Splitting the text into 1000-character chunks
        for i in range(0, len(full_text), 1000):
            chunk = full_text[i:i + 1000]
            embedding = generate_embedding(chunk).detach().numpy()
            text_chunks.append((i // 1000 + 1, chunk, embedding))  # (chunk index, text chunk)

    df = pd.DataFrame(text_chunks, columns=['Chunk_Index', 'Text_Chunk', 'Vector_Embedding'])
    return df

if(os.path.exists('patent_dataframe.pkl')):
    chunks = pd.read_pickle('patent_dataframe.pkl')

else:
    chunks = ocr_and_chunk("monoclonal-patent.pdf")
    chunks.to_pickle('patent_dataframe.pkl')
    
prompt = "Google is an equal opportunity employer and cares a lot about inclusiveness and diversity"
similar_chunk_index = search_most_similar_embedding(prompt, chunks)
#print(f"Index: {similar_chunk_index}")
similar_text = chunks[chunks['Chunk_Index'] == similar_chunk_index]['Text_Chunk'].iloc[0]
print(similar_text)