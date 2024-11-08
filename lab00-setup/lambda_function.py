import json
import uuid
import boto3
from botocore.config import Config
import os
import urllib3
import zlib

# Initialize clients
boto_config = Config(
    connect_timeout=1, read_timeout=300,
    retries={'max_attempts': 1})

boto_session = boto3.Session()
s3_client = boto3.client('s3')
sagemaker = boto3.client('sagemaker')
bedrock_runtime = boto_session.client(
    service_name="bedrock-runtime",
    config=boto_config
)

# Initialize urllib3
http = urllib3.PoolManager()

boto_config = Config(
    connect_timeout=1, read_timeout=300,
    retries={'max_attempts': 1})

boto_session = boto3.Session()
s3_client = boto3.client('s3')
bedrock_runtime = boto_session.client(
    service_name="bedrock-runtime",
    config=boto_config
)

# Use PlantUML server instead of local rendering
PLANTUML_SERVER = "http://www.plantuml.com/plantuml"

def get_default_bucket():
    """Get the default SageMaker bucket for the current region"""
    try:
        region = boto3.Session().region_name
        account = boto3.client('sts').get_caller_identity()['Account']
        bucket_name = f'sagemaker-{region}-{account}'
        
        # Check if bucket exists, if not create it
        try:
            s3_client.head_bucket(Bucket=bucket_name)
        except:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={
                    'LocationConstraint': region
                } if region != 'us-east-1' else {}
            )
            
        return bucket_name
    except Exception as e:
        raise Exception(f"Error getting default bucket: {str(e)}")

def encode_plantuml(plantuml_text):
    """
    Encode PlantUML text using the correct deflate + base64 encoding
    """
    # Remove @startuml and @enduml if present
    plantuml_text = plantuml_text.replace("@startuml", "").replace("@enduml", "").strip()
    
    # Add proper PlantUML markers
    plantuml_text = f"@startuml\n{plantuml_text}\n@enduml"
    
    # Compress using zlib
    compressed = zlib.compress(plantuml_text.encode('utf-8'))[2:-4]
    
    # Encode using PlantUML's modified base64
    base64_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
    encoded = ""
    for i in range(0, len(compressed), 3):
        chunk = compressed[i:i + 3]
        b = list(chunk) + [0] * (3 - len(chunk))
        b1 = b[0] >> 2
        b2 = ((b[0] & 0x3) << 4) | (b[1] >> 4)
        b3 = ((b[1] & 0xF) << 2) | (b[2] >> 6)
        b4 = b[2] & 0x3F
        encoded += base64_chars[b1] + base64_chars[b2]
        if len(chunk) > 1:
            encoded += base64_chars[b3]
        if len(chunk) > 2:
            encoded += base64_chars[b4]
    
    return encoded

def get_diagram_from_server(plantuml_code, output_format='png'):
    """
    Get diagram from PlantUML server using urllib3
    """
    try:
        encoded = encode_plantuml(plantuml_code)
        url = f"http://www.plantuml.com/plantuml/{output_format}/{encoded}"
        
        response = http.request('GET', url)
        
        if response.status == 200:
            return response.data
        else:
            raise Exception(f"Failed to get diagram: HTTP {response.status}")
    except Exception as e:
        raise Exception(f"Error getting diagram from server: {str(e)}")


code_generation = """
Your task is to generate a code snippet based on a given OpenAPI YAML file and a user query. The
YAML file will be provided in the following format:

<YAML_FILE>
{YAML_FILE}
</YAML_FILE>

The user query will be provided like this:

<USER_QUERY>
{USER_QUERY}
</USER_QUERY>

To complete this task, follow these steps:

1. Carefully read and understand the user query. Identify the specific API endpoint, HTTP method,
and any required parameters or request body mentioned in the query.

2. Analyze the provided YAML file and locate the relevant section that corresponds to the user's
query. The YAML file follows the OpenAPI specification and contains information about the API's
endpoints, methods, parameters, request/response bodies, etc.

3. Once you have identified the relevant section in the YAML file, extract the necessary information
to construct the code snippet, such as the endpoint URL, HTTP method, required parameters, and
request body structure (if applicable).

4. Use the extracted information to generate the code snippet in the programming language specified
by the user (e.g., Python, JavaScript, etc.). The code snippet should include the necessary
imports/libraries, the API endpoint URL, the HTTP method, any required parameters, and the request
body (if applicable).

5. Format the generated code snippet using markdown code blocks, like this:

```<programming_language>
<code_snippet>
```

This will allow the user to easily copy and paste the code snippet.

6. If the user's query cannot be fulfilled based on the provided YAML file, or if there is any
ambiguity or missing information, explain the issue politely and ask for clarification from the
user.

Remember, your goal is to generate a code snippet that accurately reflects the user's query based on
the information provided in the OpenAPI YAML file. Do not include any additional functionality or
code beyond what is necessary to fulfill the user's request.
"""


uml_generation = """
Your task is to generate a UML Sequential diagram in PlantUML format based on the given OpenAPI YAML file. The YAML file will be provided in the following format:

<YAML_FILE>
{YAML_FILE}
</YAML_FILE>

Important Instructions:
1. Analyze the YAML file carefully to understand the API endpoints, methods, and their interactions
2. Create a sequence diagram that shows:
   - Client interactions with the API endpoints
   - Request/response flows
   - Different HTTP methods used
   - Success and error scenarios
3. Your response MUST contain the PlantUML code wrapped in plantuml code blocks like this:
```plantuml
@startuml
... your sequence diagram code here ...
@enduml
```

4. Make sure to:
   - Include all API endpoints from the YAML
   - Show proper sequence flows
   - Include HTTP methods (GET, POST, etc.)
   - Show request/response data where relevant
   - Use proper PlantUML sequence diagram syntax

Generate ONLY the PlantUML code without any additional explanation or text.
"""

# Add S3 client for storing generated diagrams
s3_client = boto3.client('s3')

def get_named_parameter(event, name, default=None):
    """
    Get a parameter from the lambda event
    Args:
        event: Lambda event
        name: Parameter name
        default: Default value if parameter not found (optional)
    Returns:
        Parameter value or default if specified
    """
    try:
        return next(item for item in event['parameters'] if item['name'] == name)['value']
    except StopIteration:
        if default is not None:
            return default
        raise ValueError(f"Parameter '{name}' not found in event")

def extract_plantuml_code(response_text):
    """
    Extract PlantUML code from the response text with better error handling
    """
    try:
        # First try to find code between ```plantuml and ``` markers
        start_markers = ["```plantuml", "@startuml"]
        end_markers = ["```", "@enduml"]
        
        for start_marker in start_markers:
            if start_marker in response_text:
                start_idx = response_text.index(start_marker)
                # If we found ```plantuml, we need to skip over it
                if start_marker == "```plantuml":
                    start_idx += len(start_marker)
                
                # Find the corresponding end marker
                for end_marker in end_markers:
                    try:
                        end_idx = response_text.index(end_marker, start_idx)
                        code = response_text[start_idx:end_idx].strip()
                        
                        # Ensure the code has the required @startuml and @enduml tags
                        if not code.startswith("@startuml"):
                            code = "@startuml\n" + code
                        if not code.endswith("@enduml"):
                            code = code + "\n@enduml"
                            
                        return code
                    except ValueError:
                        continue
                        
        # If we get here, we couldn't find valid markers
        raise ValueError("No valid PlantUML code markers found")
        
    except Exception as e:
        raise ValueError(f"Could not extract PlantUML code: {str(e)}")

def generate_diagram(plantuml_code, output_format='png'):
    """
    Generate diagram using Plantweb
    """
    try:
        # Create a temporary directory to store the files
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, 'input.puml')
            output_file = os.path.join(temp_dir, f'output.{output_format}')
            
            # Write PlantUML code to temporary file
            with open(input_file, 'w') as f:
                f.write(plantuml_code)
            
            # Generate diagram
            render(input_file, output_file)
            
            # Read the generated diagram
            with open(output_file, 'rb') as f:
                diagram_data = f.read()
            
            return diagram_data
    except Exception as e:
        raise Exception(f"Error generating diagram: {str(e)}")

def upload_to_s3(diagram_data, key):
    """
    Upload generated diagram to S3 using default SageMaker bucket and return S3 URI
    """
    try:
        bucket_name = get_default_bucket()
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=diagram_data,
            ContentType='image/png'
        )
        
        # Generate S3 URI instead of presigned URL
        s3_uri = f"s3://{bucket_name}/{key}"
        
        return s3_uri
    except Exception as e:
        raise Exception(f"Error uploading to S3: {str(e)}")

def get_uml_diagram(yml_code, output_format='png'):
    try:
        content = []
        prompt = uml_generation.replace("{YAML_FILE}", yml_code)
        query_obj = {"type": "text", "text": prompt}
        content.append(query_obj)

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        })
        
        response = bedrock_runtime.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=body)
        
        response_body = json.loads(response.get("body").read())
        plantuml_text = response_body["content"][0]["text"]
        
        print(f"Claude Response: {plantuml_text}")
        
        plantuml_code = extract_plantuml_code(plantuml_text)
        print(f"Extracted PlantUML code: {plantuml_code}")
        
        # Get diagram from PlantUML server
        diagram_data = get_diagram_from_server(plantuml_code, output_format)
        
        # Upload to S3 with a unique filename
        file_name = f"diagrams/{uuid.uuid4()}.{output_format}"
        diagram_s3_uri = upload_to_s3(diagram_data, file_name)
        
        return {
            "codeBody": plantuml_code,
            "diagramUri": diagram_s3_uri  # Changed from diagramUrl to diagramUri
        }
        
    except Exception as e:
        logger.error(f"Error in get_uml_diagram: {str(e)}")
        return {
            "error": str(e),
            "codeBody": None,
            "diagramUri": None  # Changed from diagramUrl to diagramUri
        }

def get_unit_test_code(yml_code, query):
    """
    Generate unit test code to invoke an API
    
    """
    content = []

    prompt = code_generation.replace("{YAML_FILE}",yml_code)
    prompt = prompt.replace("{USER_QUERY}",query)

    query_obj = {"type": "text", "text": prompt}

    content.append(query_obj)

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
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
    
    return {"codeBody": response_body["content"][0]["text"]}

def lambda_handler(event, context):
    actionGroup = event.get('actionGroup', '')
    function = event.get('function', '')
    #parameters = event.get('parameters', [])

    try:
        if function == 'get_uml_diagram':
            yml_code = get_named_parameter(event, "yml_body")
            output_format = get_named_parameter(event, "output_format", "png")
            
            if yml_code:
                response = get_uml_diagram(yml_code, output_format)
                responseBody = {'TEXT': {'body': json.dumps(response)}}
            else:
                responseBody = {'TEXT': {'body': 'Missing YML code.'}}

        elif function == 'get_unit_test_code':
            yml_code = get_named_parameter(event, "yml_body")
            user_query = get_named_parameter(event, "user_query")

            if yml_code:
                response = get_unit_test_code(yml_code, user_query)
                responseBody = {'TEXT': {'body': json.dumps(response)}}
            else:
                responseBody = {'TEXT': {'body': 'Missing YML code.'}}

        else:
            responseBody = {'TEXT': {'body': 'Invalid function'}}

        action_response = {
            'actionGroup': actionGroup,
            'function': function,
            'functionResponse': {
                'responseBody': responseBody
            }
        }

        return {
            'response': action_response,
            'messageVersion': event['messageVersion']
        }

    except Exception as e:
        return {
            'response': {
                'actionGroup': actionGroup,
                'function': function,
                'functionResponse': {
                    'responseBody': {
                        'TEXT': {
                            'body': json.dumps({
                                'error': str(e)
                            })
                        }
                    }
                }
            },
            'messageVersion': event['messageVersion']
        }