import json
import os
import boto3
import faiss
from functools import cmp_to_key
import numpy as np
from numpy import dot
from numpy.linalg import norm
from PIL import Image
from pathlib import Path
from termcolor import colored
from lib import frames
from lib import util

TITAN_MODEL_ID = 'amazon.titan-embed-image-v1'
TITAN_PRICING = 0.00006

def batch_generate_embeddings(jpeg_files, output_dir = ''):

    output_file = os.path.join(output_dir, 'frame_embeddings.json')
    if os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as f:        
            frame_embeddings = json.load(f)
            return frame_embeddings

    frame_embeddings = []

    titan_model_id = TITAN_MODEL_ID
    accept = 'application/json'
    content_type = 'application/json'

    bedrock_runtime_client = boto3.client(service_name='bedrock-runtime')

    for jpeg_file in jpeg_files:
        with Image.open(jpeg_file) as image:
            input_image = frames.image_to_base64(image)

        model_params = {
            'inputImage': input_image,
            'embeddingConfig': {
                'outputEmbeddingLength': 384 #1024 #384 #256
            }
        }

        body = json.dumps(model_params)

        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=titan_model_id,
            accept=accept,
            contentType=content_type
        )
        response_body = json.loads(response.get('body').read())

        frame_no = int(Path(jpeg_file).stem.split('.')[1]) - 1
        frame_embeddings.append({
            'file': jpeg_file,
            'frame_no': frame_no,
            'embedding': response_body['embedding']
        })

    util.save_to_file(output_file, frame_embeddings)
    return frame_embeddings

def display_embedding_cost(frame_embeddings, display=True):
    per_image_embedding = TITAN_PRICING
    estimated_cost = per_image_embedding * len(frame_embeddings)

    if display:
        print('\n')
        print('========================================================================')
        print('Estimated cost:', colored(f"${round(estimated_cost, 4)}", 'green'), f"in us-east-1 region with {len(frame_embeddings)} embeddings")
        print('========================================================================')

    return {
        'per_image_embedding': per_image_embedding,
        'estimated_cost': estimated_cost,
        'num_embeddings': len(frame_embeddings)
    }

def create_index(dimension):
    index = faiss.IndexFlatIP(dimension) # cosine similarity
    return index

def index_frames(index, frame_embeddings):
    for item in frame_embeddings:
        embedding = np.array([item['embedding']])
        index.add(embedding)
    return index

def cosine_similarity(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim

# ascending sort by the starting shot_id first and then descending sort by the ending shot_id
def cmp_min_max(a, b):
    if a[0] < b[0]:
        return -1
    if a[0] > b[0]:
        return 1
    return b[1] - a[1]

def search_similarity(index, frame, k = 20, min_similarity = 0.80, time_range = 30):
    idx = int(frame['frame_no'])

    embedding = np.array([frame['embedding']])

    D, I = index.search(embedding, k)

    similar_frames = [
        {
            'idx': int(i),
            'similarity': float(d)
        } for i, d in zip(I[0], D[0])
    ]

    # filter out lower similiarity
    similar_frames = list(
        filter(
            lambda x: x['similarity'] > min_similarity,
            similar_frames
        )
    )

    similar_frames = sorted(similar_frames, key=lambda x: x['idx'])

    # filter out frames that are far apart from the current frame idx
    filtered_by_time_range = [{
        'idx': idx,
        'similarity': 1.0
    }]
    # filtered_by_time_range = [similar_frames[0]]

    for i in range(0, len(similar_frames)):
        prev = filtered_by_time_range[-1]
        cur = similar_frames[i]

        if abs(prev['idx'] - cur['idx']) < time_range:
               filtered_by_time_range.append(cur)

    return filtered_by_time_range



