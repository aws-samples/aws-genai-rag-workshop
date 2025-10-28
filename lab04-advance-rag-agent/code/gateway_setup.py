"""
Gateway setup script for UML generation
"""

from bedrock_agentcore_starter_toolkit.operations.gateway.client import GatewayClient
import boto3
import json
import logging
import time

def setup_gateway(lambda_arn, gateway_name="uml-generator-gateway-abc"):
    """Create and configure AgentCore Gateway"""
    region = boto3.Session().region_name
    
    # Generate unique names based on gateway name
    auth_server_name = gateway_name.replace("gateway", "auth").replace("-", "").title()
    target_name = gateway_name.replace("gateway", "target")

    print("🚀 Setting up new AgentCore Gateway for UML generation...")
    print(f"Region: {region}")
    print(f"Gateway Name: {gateway_name}")
    print(f"Auth Server: {auth_server_name}")
    print(f"Target Name: {target_name}")
    print(f"Lambda ARN: {lambda_arn}\n")

    # Initialize client
    client = GatewayClient(region_name=region)
    client.logger.setLevel(logging.INFO)

    # Step 1: Create OAuth authorizer
    print("Step 1: Creating OAuth authorization server...")
    cognito_response = client.create_oauth_authorizer_with_cognito(auth_server_name)
    print("✓ Authorization server created\n")

    # Step 2: Create Gateway
    print("Step 2: Creating Gateway...")
    gateway = client.create_mcp_gateway(
        name=gateway_name,
        role_arn=None,
        authorizer_config=cognito_response["authorizer_config"],
        enable_semantic_search=True,
    )
    print(f"✓ Gateway created: {gateway['gatewayUrl']}\n")

    # Step 3: Fix IAM permissions
    client.fix_iam_permissions(gateway)
    print("⏳ Waiting 30s for IAM propagation...")
    time.sleep(30)
    print("✓ IAM permissions configured\n")

    # Step 4: Add Lambda target
    print("Step 4: Adding UML Lambda target...")
    lambda_target = client.create_mcp_gateway_target(
        gateway=gateway,
        name=target_name,
        target_type="lambda",
        target_payload={
            "lambdaArn": lambda_arn,
            "toolSchema": {
                "inlinePayload": [
                    {
                        "name": "get_uml_diagram",
                        "description": "Generate a UML flow diagram referencing the OpenAPI API specification",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "yml_body": {
                                    "type": "string",
                                    "description": "OpenAPI standard YAML file of the swagger API"
                                }
                            },
                            "required": ["yml_body"]
                        },
                        "outputSchema": {
                            "type": "object",
                            "properties": {
                                "diagramUri": {
                                    "type": "string",
                                    "description": "S3 URI of the generated diagram"
                                },
                                "codeBody": {
                                    "type": "string", 
                                    "description": "PlantUML code for the diagram"
                                }
                            }
                        }
                    }
                ]
            }
        },
        credentials=None,
    )
    print("✓ UML Lambda target added\n")

    # Step 5: Save configuration
    config = {
        "gateway_url": gateway["gatewayUrl"],
        "gateway_id": gateway["gatewayId"],
        "region": region,
        "client_info": cognito_response["client_info"]
    }

    with open("gateway_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print("✅ Gateway setup complete!")
    print(f"Gateway URL: {gateway['gatewayUrl']}")
    print(f"Gateway ID: {gateway['gatewayId']}")
    print("\nConfiguration saved to: gateway_config.json")
    print("=" * 60)

    return config
