import os
import json
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

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

    similarities = []

    for index, row in df.iterrows():
        chunk_embedding = torch.tensor(row['Vector_Embedding'], dtype=torch.float)
        #chunk_val = row['Text_Chunk']
        similarity = cos_sim(prompt_embedding, chunk_embedding)
        #similarity = torch.nn.functional.cosine_similarity(prompt_embedding, chunk_embedding, dim=0)
        #print(f"Chunk Text: {chunk_val}, Similarity: {similarity}")
        similarities.append((index, similarity.item()))

    similarities.sort(key=lambda x: x[1], reverse=True)
    sorted_indices = [index for index, _ in similarities]

    return sorted_indices

        #if similarity > max_similarity:
        #    max_similarity = similarity
        #    most_similar_index = row['Chunk_Index']


def build_dataframe_old(chunks):
    """For older reducto output format"""

    chunk_index = 0
    text_chunks = []

    for chunk in chunks:
        if 'chunk_table_plaintext' in chunk:
            chunk_val = chunk['chunk_val'] + "\n" + chunk['chunk_table_plaintext'] 
            embedding = generate_embedding(chunk_val).detach().numpy()
            #embedding = generate_embedding(chunk['chunk_table_plaintext']).detach().numpy()
        else:
            embedding = generate_embedding(chunk['chunk_val']).detach().numpy()

        chunk_text = chunk['chunk_val']

        for field, value in chunk['chunk_metadata'].items():
            metadata_field = f"{field}: {value}"
            chunk_text += f"\n {metadata_field}"

        text_chunks.append((chunk_index, chunk_text, embedding))

        chunk_index += 1

    df = pd.DataFrame(text_chunks, columns=['Chunk_Index', 'Text_Chunk', 'Vector_Embedding'])
    return df

def build_dataframe(chunks):

    chunk_index = 0
    text_chunks = []

    for chunk in chunks:
        embedding = generate_embedding(chunk['embed']).detach().numpy()

        chunk_text = chunk['raw_text']

        if 'title' in chunk:
            chunk_text += f"\n Title: {chunk['title']}"

        if 'section_header' in chunk:
            chunk_text += f"\n Section Header: {chunk['section_header']}"

        text_chunks.append((chunk_index, chunk_text, embedding))

        chunk_index += 1

    df = pd.DataFrame(text_chunks, columns=['Chunk_Index', 'Text_Chunk', 'Vector_Embedding'])
    return df

if __name__ == "__main__":

    with open('monoclonal_patent.json', 'r') as file:
        data = json.load(file)

    if(os.path.exists('reducto_patent_dataframe.pkl')):
        chunks = pd.read_pickle('reducto_patent_dataframe.pkl')

    else:
        chunks = build_dataframe(data)
        chunks.to_pickle('reducto_patent_dataframe.pkl')