import json
import boto3
from botocore.config import Config
from botocore.exceptions import NoCredentialsError
import io
import base64
from PIL import Image
import random
import uuid

boto_config = Config(
        connect_timeout=1, read_timeout=300,
        retries={'max_attempts': 1})

boto_session = boto3.Session()

bedrock_runtime = boto_session.client(
    service_name="bedrock-runtime",
    config=boto_config
)

s3 = boto_session.client('s3')


# load video meta data
def load_json_to_dict(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


# check image size and down size
def compress_image(image, max_size=5242880):  # 5 MB in bytes
    # Get the current file size    12582912
    current_size = image.size[0] * image.size[1] * 3
    
    if current_size > max_size:
        # Calculate the compression ratio
        compression_ratio = max_size / current_size
        
        # Calculate the new dimensions
        new_width = int(image.size[0] * compression_ratio ** 0.5)
        new_height = int(image.size[1] * compression_ratio ** 0.5)
        
        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    else:
        resized_image = image
        
    return resized_image


# download an image file from S3
def download_file_from_s3(s3_file_path):

    # Split the S3 file path into bucket name and file key
    bucket_name, file_key = s3_file_path.replace("s3://", "").split('/', 1)
    
    try:
        # Download the file from S3
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        image_data = response['Body'].read()
        image = Image.open(io.BytesIO(image_data))

        image = image.convert("RGB")
        image = compress_image(image)

        return image
    except Exception as e:
        print(f"Error downloading file '{s3_file_path}': {e}")
        return None


# get embeddings from bedrock
def get_mm_embedding(image_base64=None, text_description=None, model_id="amazon.titan-embed-image-v1"):
    input_data = {}

    if image_base64 is not None:
        input_data["inputImage"] = image_base64
    if text_description is not None:
        input_data["inputText"] = text_description

    if not input_data:
        raise ValueError("At least one of image_base64 or text_description must be provided")

    body = json.dumps(input_data)

    response = bedrock_runtime.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get("body").read())
    return response_body.get("embedding")


# get text embeddings from bedrock
def get_text_embedding(text, model_id="cohere.embed-english-v3"):
    input_data = {}

    if text is None:
        raise ValueError("Text cannot by None.")

    if model_id == "amazon.titan-embed-text-v1":
        input_data["inputText"] = text
    elif model_id == "amazon.titan-embed-text-v2:0":
        input_data["inputText"] = text
        input_data["dimensions"] = 1024
        input_data["normalize"] = False
        
    else: # Cohere
        input_data["texts"] = [text]
        input_data["input_type"] = "search_document"

    body = json.dumps(input_data)

    response = bedrock_runtime.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get("body").read())

    if model_id == "amazon.titan-embed-text-v1":
        return response_body.get('embedding')
    elif model_id == "amazon.titan-embed-text-v2:0":
        return response_body["embedding"]
    else: # Cohere
        return response_body["embeddings"][0]


# encode image to base64
def _encode(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode('utf-8')


# Generate an image using SDXL 1.0 on demand 
def get_image_response(json_body):
   
    accept = "application/json"
    content_type = "application/json"

    model_id='stability.stable-diffusion-xl-v1'

    response = bedrock_runtime.invoke_model(
        body=json.dumps(json_body), modelId=model_id, accept=accept, contentType=content_type
    )
    response_body = json.loads(response.get("body").read())
    print(response_body['result'])

    base64_image = response_body.get("artifacts")[0].get("base64")
    base64_bytes = base64_image.encode('ascii')
    image_bytes = base64.b64decode(base64_bytes)

    image = Image.open(io.BytesIO(image_bytes))

    finish_reason = response_body.get("artifacts")[0].get("finishReason")

    if finish_reason == 'ERROR' or finish_reason == 'CONTENT_FILTERED':
        raise f"Image generation error. Error code is {finish_reason}"

    return image


# generate text response
def get_text_response(image_base64=None, text_query="What is in this image?"):

    content = []

    img_obj = dict()
    query_obj = {"type": "text", "text": text_query}
        
    if image_base64:
        img_obj["type"] = "image"
        img_obj["source"] = {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": image_base64,
        }
        content.append(img_obj)

    content.append(query_obj)

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 10000,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }
    )

    response = bedrock_runtime.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body=body)
    
    response_body = json.loads(response.get("body").read())

    return response_body


# calculate the token price
def calc_total_cost(price, tokens):
    return sum(price.get(key, 0) * tokens.get(key, 0) for key in set(price) | set(tokens))


# load jsonl from file
def load_jsonl(file_path):
    """
    Loads a JSONL file into a list of dictionaries.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data
    

# split data into train and validation
def random_split(data, pct=0.8):

    random.shuffle(data)

    # Calculate the split indices
    num_training = int(len(data) * pct)

    # Split the data into training and validation sets
    training = data[:num_training]
    validation = data[num_training:]

    training_data = dict()
    
    for d in training:
        image_id = str(uuid.uuid4())
        training_data[image_id] = d

    validation_data = dict()
    for d in validation:
        image_id = str(uuid.uuid4())
        validation_data[image_id] = d

    return training_data, validation_data


#evaluate top hits
def evaluate_top_hit(os_manager, os_query, dataset, index_name, top_k=5, model_id=None):
    queries = dataset["queries"]
    mapping = dataset["relevant_docs"]
    eval_results = []
    for q_id, query in queries.items():

        if model_id == "amazon.titan-embed-text-v2:0":
            os_query["query"]["knn"]["vector_field"]["vector"] = get_text_embedding(query, model_id="amazon.titan-embed-text-v2:0")
        else:
            os_query["query"]["knn"]["vector_field"]["vector"] = get_mm_embedding(text_description=query)
        
        os_query["size"] = top_k
        os_query["query"]["knn"]["vector_field"]["k"] = top_k
            
        results = os_manager.opensearch_query(os_query,
                                              index_name=index_name)
    
        retrieved_ids = []
        for index, value in enumerate(results):
            retrieved_ids.append(value["_source"]["id"])
    
        expected_id = mapping[q_id][0]
        
        is_hit = expected_id in retrieved_ids  # assume 1 relevant doc
    
        eval_result = {
            'is_hit': is_hit,
            'retrieved': retrieved_ids,
            'expected': expected_id,
            'query': q_id,
        }
        eval_results.append(eval_result)
    return eval_results