import json
import uuid
import boto3
from botocore.config import Config

boto_config = Config(
        connect_timeout=1, read_timeout=300,
        retries={'max_attempts': 1})

boto_session = boto3.Session()

bedrock_runtime = boto_session.client(
    service_name="bedrock-runtime",
    config=boto_config
)

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
Your task is to generate a UML Sequential plantuml code snippet based on a given OpenAPI YAML file . The
YAML file will be provided in the following format:

<YAML_FILE>
{YAML_FILE}
</YAML_FILE>

Here are the steps you should follow:

1. Read the provided <yaml> carefully and understand the APIs, their
endpoints, request/response data structures, and other details.

<yaml>
{YAML_FILE}
</yaml>

2. In the <PlantUML> field, generate the UML Sequential Plantuml code generated after looking at YAML file. Make sure you are looking at all the elements of YAML file as you generate the plantuml code.

3. Structure your JSON output as follows:

{
"PlantUML": "<PlantUML>"
}

4. Replace <PlantUML> with the appropriate content you generated in the previous steps.

Please provide your response in JSON format only, without any additional explanations or comments.
"""

uml_generation1 = """
Your task is to generate a UML Sequential plantuml code snippet based on a given OpenAPI YAML file . The
YAML file will be provided in the following format:

<YAML_FILE>
{YAML_FILE}
</YAML_FILE>

Here are the steps you should follow:

1. Read the provided <yaml> carefully and understand the APIs, their
endpoints, request/response data structures, and other details.

<yaml>
{YAML_FILE}
</yaml>

2. Format the generated code snippet using markdown code blocks, like this:

```<programming_language>
<code_snippet>
```

This will allow the user to easily copy and paste the code snippet.


Remember, your goal is to generate a code snippet that accurately reflects input OpenAPI YAML in UML Sequential PlantUML code. Do not include any additional functionality or
code beyond what is necessary to fulfill the user's request.
"""

def get_named_parameter(event, name):
    """
    Get a parameter from the lambda event
    """
    return next(item for item in event['parameters'] if item['name'] == name)['value']

def get_uml_diagram(yml_code):
    """
    Get a UML Diagram

    Save image to S3
    """
    content = []

    prompt = uml_generation1.replace("{YAML_FILE}",yml_code)

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
    # get the action group used during the invocation of the lambda function
    actionGroup = event.get('actionGroup', '')
    
    # name of the function that should be invoked
    function = event.get('function', '')
    
    # parameters to invoke function with
    parameters = event.get('parameters', [])

    if function == 'get_uml_diagram':
        yml_code = get_named_parameter(event, "yml_body")
        
        if yml_code:
            response = get_uml_diagram(yml_code)
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

    function_response = {'response': action_response, 'messageVersion': event['messageVersion']}
    print("Response: {}".format(function_response))

    return function_response
