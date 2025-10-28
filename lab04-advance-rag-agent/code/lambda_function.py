import json
import boto3
import uuid
import os
from datetime import datetime
import logging
import urllib3
import zlib

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'swagger-diagrams-bucket-1730133000')

# Initialize urllib3
http = urllib3.PoolManager()

def encode_plantuml(plantuml_text):
    """Encode PlantUML text using the correct deflate + base64 encoding"""
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
        chunk = compressed[i:i+3]
        if len(chunk) == 3:
            b1, b2, b3 = chunk[0], chunk[1], chunk[2]
            encoded += base64_chars[b1 >> 2]
            encoded += base64_chars[((b1 & 0x03) << 4) | (b2 >> 4)]
            encoded += base64_chars[((b2 & 0x0F) << 2) | (b3 >> 6)]
            encoded += base64_chars[b3 & 0x3F]
        elif len(chunk) == 2:
            b1, b2 = chunk[0], chunk[1]
            encoded += base64_chars[b1 >> 2]
            encoded += base64_chars[((b1 & 0x03) << 4) | (b2 >> 4)]
            encoded += base64_chars[(b2 & 0x0F) << 2]
        elif len(chunk) == 1:
            b1 = chunk[0]
            encoded += base64_chars[b1 >> 2]
            encoded += base64_chars[(b1 & 0x03) << 4]
    
    return encoded

def get_diagram_from_server(plantuml_code, output_format='png'):
    """Get diagram from PlantUML server using urllib3"""
    try:
        encoded = encode_plantuml(plantuml_code)
        url = f"http://www.plantuml.com/plantuml/{output_format}/{encoded}"
        
        response = http.request('GET', url)
        
        if response.status == 200:
            return response.data
        else:
            raise Exception(f"PlantUML server returned status {response.status}")
            
    except Exception as e:
        logger.error(f"Error getting diagram from server: {str(e)}")
        raise

def lambda_handler(event, context):
    """Generate UML diagram from OpenAPI YAML specification"""
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        yml_body = event.get('yml_body', '')
        
        if not yml_body:
            logger.error(f"Missing yml_body parameter. Event: {event}")
            return {
                'diagramUri': 'Not available',
                'codeBody': 'Error: No YAML content provided'
            }
        
        logger.info(f"Processing YAML body of length: {len(yml_body)}")
        
        # Generate PlantUML code from OpenAPI spec
        plantuml_code = generate_plantuml_from_openapi(yml_body)
        
        # Generate PNG image from PlantUML server
        logger.info("Generating PNG image from PlantUML server...")
        diagram_data = get_diagram_from_server(plantuml_code, 'png')
        
        # Upload PNG image to S3
        diagram_id = str(uuid.uuid4())
        s3_key = f"diagrams/{diagram_id}.png"
        
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=diagram_data,
            ContentType='image/png'
        )
        
        s3_uri = f"s3://{BUCKET_NAME}/{s3_key}"
        logger.info(f"Successfully uploaded PNG diagram to: {s3_uri}")
        
        return {
            'diagramUri': s3_uri,
            'codeBody': plantuml_code
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return {
            'diagramUri': 'Not available',
            'codeBody': f'Error generating diagram: {str(e)}'
        }

def generate_plantuml_from_openapi(yml_content):
    """Generate PlantUML code from OpenAPI YAML content"""
    
    plantuml = "@startuml\n"
    plantuml += "!theme plain\n"
    plantuml += "title API Flow Diagram\n\n"
    
    lines = yml_content.split('\n')
    
    # Extract API info
    api_title = "API"
    for line in lines:
        if 'title:' in line:
            api_title = line.split('title:')[1].strip().strip('"\'')
            break
    
    # Add main API component
    plantuml += f"participant Client\n"
    plantuml += f"participant \"{api_title}\" as API\n"
    plantuml += f"database \"Data Store\" as DB\n\n"
    
    # Extract paths and generate sequence
    in_paths = False
    current_path = None
    
    for line in lines:
        line = line.strip()
        
        if line == 'paths:':
            in_paths = True
            continue
            
        if in_paths and line.startswith('/'):
            current_path = line.rstrip(':')
            plantuml += f"== {current_path} ==\n"
            
        elif in_paths and line in ['get:', 'post:', 'put:', 'delete:', 'patch:']:
            method = line.rstrip(':').upper()
            plantuml += f"Client -> API: {method} {current_path}\n"
            plantuml += f"API -> DB: Query/Update\n"
            plantuml += f"DB -> API: Response\n"
            plantuml += f"API -> Client: HTTP Response\n\n"
    
    # Add data models section
    plantuml += "== Data Models ==\n"
    plantuml += "note over API\n"
    plantuml += "Data Models:\n"
    
    # Extract schema info
    in_components = False
    for line in lines:
        line = line.strip()
        if line == 'components:' or line == 'definitions:':
            in_components = True
            continue
        if in_components and line.endswith(':') and not line.startswith(' '):
            schema_name = line.rstrip(':')
            if schema_name not in ['schemas', 'properties', 'type', 'required']:
                plantuml += f"- {schema_name}\n"
    
    plantuml += "end note\n"
    plantuml += "@enduml"
    
    return plantuml
