import json
import boto3
from botocore.config import Config
from PIL import Image, ImageDraw
import copy
from io import BytesIO
import base64
import yaml

boto_config = Config(
        connect_timeout=1, read_timeout=300,
        retries={'max_attempts': 1})

boto_session = boto3.Session()

bedrock_runtime = boto_session.client(
    service_name="bedrock-runtime",
    config=boto_config
)

s3 = boto_session.client('s3')

# upload a file to S3
def upload_file_to_s3(file_name, bucket_name, s3_key, metadata={}):
    """
    Uploads a local file to an Amazon S3 bucket.

    Args:
        file_name (str): The name of the local file to upload.
        bucket_name (str): The name of the S3 bucket to upload the file to.
        s3_key (str): The key (file path) to use for the uploaded file in the S3 bucket.

    Returns:
        str: The full S3 path of the uploaded file (e.g., s3://bucket_name/s3_key).
    """

    try:
        s3.upload_file(file_name, 
                       bucket_name, 
                       s3_key, 
                       ExtraArgs={
                            'Metadata': metadata
                        }
                      )
        print(f"File '{file_name}' uploaded successfully to s3://{bucket_name}/{s3_key}")
        s3_full_path = f"s3://{bucket_name}/{s3_key}"
        return s3_full_path
    except Exception as e:
        print(f"Error uploading file '{file_name}' to S3: {e}")
        return None

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



def load_yaml_to_string(file_path):
    """
    Load a YAML file into a string.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        str: The contents of the YAML file as a string.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        yaml_data = yaml.safe_load(file)
        yaml_string = yaml.dump(yaml_data, 
                                default_flow_style=False, 
                                sort_keys=False)
    return yaml_string

def image_to_base64(image):
    buff = BytesIO()
    image.save(buff, format='JPEG')
    return base64.b64encode(buff.getvalue()).decode('utf8')