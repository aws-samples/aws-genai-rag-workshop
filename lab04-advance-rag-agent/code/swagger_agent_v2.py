import boto3
import json
import requests
import os
from datetime import datetime
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent, tool
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamablehttp_client
from bedrock_agentcore_starter_toolkit.operations.gateway.client import GatewayClient

# Configuration
REGION_NAME = boto3.Session().region_name
BEDROCK_KB_ID = os.environ.get('BEDROCK_KB_ID', 'MRQAJK5CPC')

# Initialize clients
bedrock_agent_client = boto3.client('bedrock-agent-runtime', region_name=REGION_NAME)
bedrock_runtime = boto3.client('bedrock-runtime', region_name=REGION_NAME)

def create_streamable_http_transport(mcp_url: str, access_token: str):
    return streamablehttp_client(mcp_url, headers={"Authorization": f"Bearer {access_token}"})

def get_full_tools_list(client):
    """List tools w/ support for pagination"""
    more_tools = True
    tools = []
    pagination_token = None
    while more_tools:
        tmp_tools = client.list_tools_sync(pagination_token=pagination_token)
        tools.extend(tmp_tools)
        if tmp_tools.pagination_token is None:
            more_tools = False
        else:
            more_tools = True 
            pagination_token = tmp_tools.pagination_token
    return tools

def create_mcp_client():
    """Create MCP client with new Gateway authentication"""
    try:
        with open('new_gateway_config.json', 'r') as f:
            config = json.load(f)
        
        # Get access token using GatewayClient
        gateway_client = GatewayClient(region_name=config["region"])
        access_token = gateway_client.get_access_token_for_cognito(config["client_info"])
        
        return MCPClient(lambda: create_streamable_http_transport(
            config['gateway_url'], 
            access_token
        ))
    except Exception as e:
        print(f"Error creating MCP client: {e}")
        return None

# Initialize MCP client
mcp_client = create_mcp_client()
if mcp_client:
    print("MCP client initialized successfully")
else:
    print("MCP client initialization failed")
    mcp_client = None

# Define tools
@tool
def retrieve_swagger_spec(query: str) -> str:
    """Retrieve OpenAPI/Swagger specification from the knowledge base.
    
    Args:
        query: search query to find relevant API specification
    """
    try:
        response = bedrock_agent_client.retrieve(
            knowledgeBaseId=BEDROCK_KB_ID,
            retrievalQuery={'text': query},
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': 5
                }
            }
        )
        
        results = []
        for result in response.get('retrievalResults', []):
            content = result.get('content', {}).get('text', '')
            if content:
                results.append(content)
        
        if results:
            return f"Found {len(results)} relevant API specifications:\n\n" + "\n\n---\n\n".join(results)
        else:
            return "No relevant API specifications found in the knowledge base."
            
    except Exception as e:
        return f"Error retrieving from knowledge base: {str(e)}"

@tool
def get_unit_test_code(yml_body: str, user_query: str) -> str:
    """Generate functional testing code referencing the OpenAPI API specification.
    
    Args:
        yml_body: openapi standard yml file of the swagger api
        user_query: question from user or code user wants to generate
    """
    try:
        prompt = f"""Generate Python unit test code for the following OpenAPI specification:

{yml_body}

User request: {user_query}

Please generate comprehensive unit tests using pytest or unittest framework that cover:
1. All API endpoints defined in the specification
2. Different HTTP methods (GET, POST, PUT, DELETE)
3. Request/response validation
4. Error handling scenarios
5. Mock responses for testing

Return only the Python test code with proper imports and structure."""

        response = bedrock_runtime.converse(
            modelId="anthropic.claude-3-7-sonnet-20250219-v1:0",
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ],
            inferenceConfig={
                "maxTokens": 4000,
                "temperature": 0.1
            }
        )
        
        return response['output']['message']['content'][0]['text']
        
    except Exception as e:
        return f"Error generating unit test code: {str(e)}"

@tool
def get_uml_diagram(yml_body: str) -> str:
    """Generate a UML flow diagram referencing the OpenAPI API specification.
    
    Args:
        yml_body: openapi standard yml file of the swagger api
    """
    try:
        if mcp_client is None:
            return "Gateway MCP client not available. UML diagram generation requires Gateway connection."
        
        with mcp_client:
            # Find the UML generator tool dynamically
            tools = get_full_tools_list(mcp_client)
            uml_tool_name = None
            for tool in tools:
                if "get_uml_diagram" in tool.tool_name:
                    uml_tool_name = tool.tool_name
                    break
            
            if not uml_tool_name:
                return "UML generator tool not found in Gateway."
            
            # Use the dynamically found gateway tool
            result = mcp_client.call_tool_sync(
                tool_use_id="uml-diagram-request",
                name=uml_tool_name,
                arguments={"yml_body": yml_body}
            )
            
            # Process the result
            if result and result.get('status') == 'success':
                content = result.get('content', [])
                if content and len(content) > 0:
                    response_text = content[0].get('text', str(result))
                    try:
                        parsed = json.loads(response_text)
                        s3_uri = parsed.get('diagramUri', 'Not available')
                        plantuml_code = parsed.get('codeBody', 'Not available')
                        
                        return f"""UML Diagram Generated Successfully!

S3 Location: {s3_uri}

PlantUML Code:
{plantuml_code}

The diagram has been generated and saved as a PNG image to S3. You can download and view the visual diagram from the S3 location above."""
                    except json.JSONDecodeError:
                        return f"UML diagram generated. Response: {response_text}"
                else:
                    return "UML diagram tool executed but returned no content."
            else:
                return f"UML diagram generation failed. Result: {result}"
                
    except Exception as e:
        return f"Error generating UML diagram: {str(e)}"

# Create the agent
def create_agent():
    """Create the Strands agent with all tools"""
    model = BedrockModel(
        model_id="anthropic.claude-3-7-sonnet-20250219-v1:0",
        streaming=True,
    )
    
    tools = [retrieve_swagger_spec, get_unit_test_code, get_uml_diagram]
    
    return Agent(
        model=model,
        tools=tools,
        system_prompt="""You are a helpful assistant specialized in working with OpenAPI/Swagger specifications. 

You have access to:
1. A knowledge base containing API specifications
2. Tools to generate unit test code for APIs
3. Tools to generate UML diagrams for APIs

When users ask about APIs:
- First retrieve relevant specifications from the knowledge base
- Generate UML diagrams to visualize API structure
- Create unit tests when requested
- Provide clear explanations of API functionality

Always be helpful and provide comprehensive responses with examples when appropriate."""
    )

# Initialize the agent
agent = create_agent()

# Create the AgentCore app
app = BedrockAgentCoreApp()

if __name__ == "__main__":
    print("Swagger Agent v2 initialized with new Gateway integration!")
    print("Available tools:")
    print("- retrieve_swagger_spec: Get API specs from knowledge base")
    print("- get_unit_test_code: Generate Python unit tests")
    print("- get_uml_diagram: Generate UML diagrams via Gateway")
