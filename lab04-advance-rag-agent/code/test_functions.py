"""
Testing functions for Agent v2 and Gateway
"""

from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamablehttp_client
from bedrock_agentcore_starter_toolkit.operations.gateway.client import GatewayClient
import json
import subprocess
import os
import re

def create_streamable_http_transport(mcp_url: str, access_token: str):
    return streamablehttp_client(mcp_url, headers={"Authorization": f"Bearer {access_token}"})

def get_full_tools_list(client):
    """Get all tools with pagination support"""
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

def test_gateway_direct():
    """Test Gateway functionality directly"""
    # Load gateway configuration
    with open("gateway_config.json", "r") as f:
        config = json.load(f)

    # Get access token
    print("Getting access token...")
    client = GatewayClient(region_name=config["region"])
    access_token = client.get_access_token_for_cognito(config["client_info"])
    print("✓ Access token obtained\n")

    # Setup MCP client
    mcp_client = MCPClient(lambda: create_streamable_http_transport(config["gateway_url"], access_token))

    with mcp_client:
        # List available tools
        tools = get_full_tools_list(mcp_client)
        tool_names = [tool.tool_name for tool in tools]
        print(f"📋 Available tools: {tool_names}")

        # Find the UML generator tool dynamically
        uml_tool_name = None
        for tool_name in tool_names:
            if "get_uml_diagram" in tool_name:
                uml_tool_name = tool_name
                break
        
        if not uml_tool_name:
            print("❌ UML generator tool not found!")
            return None

        print(f"🎯 Using tool: {uml_tool_name}")

        # Test UML generation
        test_yaml = """openapi: 3.0.0
info:
  title: Test API
  version: 1.0.0
paths:
  /users:
    get:
      summary: Get users
    post:
      summary: Create user"""

        print("🧪 Testing UML diagram generation...")
        result = mcp_client.call_tool_sync(
            tool_use_id="test-uml-generation",
            name=uml_tool_name,
            arguments={"yml_body": test_yaml}
        )
        
        if result.get('status') == 'success':
            content = result.get('content', [])
            if content:
                response_text = content[0].get('text', str(result))
                try:
                    parsed = json.loads(response_text)
                    print(f"📍 S3 URI: {parsed.get('diagramUri', 'Not found')}")
                    return parsed.get('diagramUri')
                except:
                    print(f"📄 Raw response: {response_text}")
        
        return None

def test_agent_comprehensive():
    """Run comprehensive agent tests"""
    # Load gateway configuration
    with open("gateway_config.json", "r") as f:
        config = json.load(f)

    # Get access token
    client = GatewayClient(region_name=config["region"])
    access_token = client.get_access_token_for_cognito(config["client_info"])

    # Setup MCP client and agent
    mcp_client = MCPClient(lambda: create_streamable_http_transport(config["gateway_url"], access_token))

    with mcp_client:
        tools = get_full_tools_list(mcp_client)
        
        model = BedrockModel(
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            streaming=True,
        )
        
        agent = Agent(model=model, tools=tools)

        # Test 1: Tool availability
        print("🧪 Test 1: Tool Availability")
        response = agent("What tools do you have available?")
        print(f"✅ Response: {response.message.get('content', [''])[0][:300]}...")

        # Test 2: Simple UML Generation
        print("\n🧪 Test 2: Simple UML Generation")
        response = agent("Generate a UML diagram for a simple API with GET /users endpoint and show me the S3 location")
        content = response.message.get("content", [""])[0]
        
        # Extract S3 URI
        s3_match = re.search(r"s3://[^\s]+", content)
        if s3_match:
            s3_uri = s3_match.group()
            print(f"📍 S3 URI: {s3_uri}")
            
            # Store for later download
            with open("test_s3_uris.txt", "a") as f:
                f.write(f"{s3_uri}\n")

        # Test 3: Complex E-commerce API UML
        print("\n🧪 Test 3: Complex E-commerce API UML")
        response = agent("""
        Generate a UML diagram for an e-commerce API that has these endpoints:
        - GET /products - list all products
        - POST /products - create a new product  
        - GET /products/{id} - get product details
        - POST /orders - create an order
        - GET /orders/{id} - get order details
        
        Show me the S3 location where the diagram is stored.
        """)
        
        content = response.message.get("content", [""])[0]
        s3_match = re.search(r"s3://[^\s]+", content)
        if s3_match:
            s3_uri = s3_match.group()
            print(f"📍 S3 URI: {s3_uri}")
            
            # Store for later download
            with open("test_s3_uris.txt", "a") as f:
                f.write(f"{s3_uri}\n")

def download_and_verify_diagrams():
    """Download generated diagrams and verify they are valid PNG files"""
    print("📥 Downloading and verifying generated UML diagrams...")
    
    # Read S3 URIs from test results
    s3_uris = []
    try:
        with open("test_s3_uris.txt", "r") as f:
            s3_uris = [line.strip() for line in f.readlines() if line.strip()]
        print(f"📋 Found {len(s3_uris)} S3 URIs to download")
    except FileNotFoundError:
        print("⚠️ No S3 URIs file found.")
        return

    # Download each diagram
    downloaded_files = []
    for i, s3_uri in enumerate(s3_uris, 1):
        if s3_uri:
            filename = f"test_uml_diagram_{i}.png"
            try:
                result = subprocess.run([
                    "aws", "s3", "cp", s3_uri, filename
                ], capture_output=True, text=True, check=True)
                
                print(f"✅ Downloaded: {filename}")
                downloaded_files.append(filename)
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to download {s3_uri}: {e.stderr}")

    # Verify PNG files
    print(f"\n🔍 Verifying {len(downloaded_files)} downloaded files...")
    for filename in downloaded_files:
        try:
            with open(filename, "rb") as f:
                header = f.read(8)
                if header == b"\x89PNG\r\n\x1a\n":
                    size = os.path.getsize(filename)
                    print(f"✅ {filename}: Valid PNG ({size:,} bytes)")
                else:
                    print(f"❌ {filename}: Invalid PNG header")
        except Exception as e:
            print(f"❌ Error verifying {filename}: {e}")

def test_agentcore_cli():
    """Test the deployed agent using AgentCore CLI"""
    print("🧪 Testing deployed Agent v2 with AgentCore CLI...")
    
    # Test 1: Simple tool discovery
    print("\n1. Testing tool discovery...")
    try:
        result = subprocess.run([
            "agentcore", "invoke", 
            json.dumps({"prompt": "What tools do you have available?"})
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Tool discovery test passed")
            if "s3://" in result.stdout:
                print("✅ S3 URI found in response")
        else:
            print(f"❌ Tool discovery test failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("❌ Tool discovery test timed out")
    except Exception as e:
        print(f"❌ Tool discovery test error: {e}")

    # Test 2: UML generation
    print("\n2. Testing UML generation...")
    try:
        result = subprocess.run([
            "agentcore", "invoke", 
            json.dumps({"prompt": "Generate a UML diagram for a simple API with GET /users endpoint and show me the S3 location"})
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ UML generation test passed")
            if "s3://" in result.stdout:
                print("✅ S3 URI found in response")
        else:
            print(f"❌ UML generation test failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("❌ UML generation test timed out")
    except Exception as e:
        print(f"❌ UML generation test error: {e}")
