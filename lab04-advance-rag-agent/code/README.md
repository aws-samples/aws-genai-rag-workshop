# Code Directory

This directory contains the core code files for the Strands Agent v2 with Gateway integration project.

## Files

### `lambda_function.py`
- **Purpose**: AWS Lambda function for UML diagram generation
- **Features**: 
  - Converts OpenAPI YAML to PlantUML code
  - Generates PNG images via PlantUML server
  - Uploads diagrams to S3 with proper metadata
- **Dependencies**: boto3, urllib3, zlib

### `swagger_agent_v2.py`
- **Purpose**: Strands Agent v2 with Gateway integration
- **Features**:
  - Knowledge base integration for API specs
  - UML diagram generation via Gateway
  - Unit test code generation
  - Complete AgentCore deployment ready
- **Dependencies**: strands-agents, boto3, bedrock-agentcore-starter-toolkit

### `gateway_setup.py`
- **Purpose**: AgentCore Gateway configuration and setup
- **Features**:
  - OAuth authorization server creation
  - Gateway creation with IAM permissions
  - Lambda target integration with tool schema
  - Configuration file generation
- **Dependencies**: bedrock-agentcore-starter-toolkit

### `test_functions.py`
- **Purpose**: Comprehensive testing and validation functions
- **Features**:
  - Direct Gateway testing
  - Agent integration testing
  - CLI testing for deployed agents
  - PNG file download and verification
- **Dependencies**: strands-agents, subprocess, boto3

## Usage

These files are referenced by the Jupyter notebook `deploy-strands-to-bedrock-agentcorev2.ipynb` to keep the notebook clean and organized. Each file can also be used independently for specific tasks.

## Architecture

```
User Query → Agent v2 → Gateway → Lambda → PlantUML Server → S3 Storage
     ↓           ↓         ↓        ↓           ↓              ↓
  Natural    Strands   OAuth    PNG Gen    UML Render    Image Store
  Language   Agent     Auth     Python     External API   AWS S3
```
