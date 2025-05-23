import json
import boto3
import random
import time
import zipfile
from io import BytesIO
import sys

suffix = random.randrange(200, 900)
boto3_session = boto3.session.Session()
region_name = boto3_session.region_name
iam_client = boto3_session.client('iam')
lambda_client = boto3_session.client('lambda')
account_number = boto3.client('sts').get_caller_identity().get('Account')
identity = boto3.client('sts').get_caller_identity()['Arn']

encryption_policy_name = f"bedrock-sample-rag-sp-{suffix}"
network_policy_name = f"bedrock-sample-rag-np-{suffix}"
access_policy_name = f'bedrock-sample-rag-ap-{suffix}'
bedrock_execution_role_name = f'AmazonBedrockExecutionRoleForKnowledgeBase_{suffix}'
fm_policy_name = f'AmazonBedrockFoundationModelPolicyForKnowledgeBase_{suffix}'
s3_policy_name = f'AmazonBedrockS3PolicyForKnowledgeBase_{suffix}'
kb_policy_name = f'AmazonBedrockAgentKnowledgeBase_{suffix}'
oss_policy_name = f'AmazonBedrockOSSPolicyForKnowledgeBase_{suffix}'
lambda_fm_policy_name = f'AWSLambdaFoundationModelPolicy_{suffix}'
lambda_function_role_name = f'AWSLambdaExecutionRole_{suffix}'

def create_bedrock_execution_role(bucket_name):
    foundation_model_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                ],
                "Resource": "*"
            }
        ]
    }

    s3_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:ListBucket"
                ],
                "Resource": [
                    f"arn:aws:s3:::{bucket_name}",
                    f"arn:aws:s3:::{bucket_name}/*"
                ],
                "Condition": {
                    "StringEquals": {
                        "aws:ResourceAccount": f"{account_number}"
                    }
                }
            }
        ]
    }

    bedrock_kb_retrival_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:Retrieve",
                    "bedrock:RetrieveAndGenerate"
                ],
                "Resource": "*"
            }
        ]
    }

    assume_role_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "bedrock.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    # create policies based on the policy documents
    fm_policy = iam_client.create_policy(
        PolicyName=fm_policy_name,
        PolicyDocument=json.dumps(foundation_model_policy_document),
        Description='Policy for accessing foundation model',
    )

    s3_policy = iam_client.create_policy(
        PolicyName=s3_policy_name,
        PolicyDocument=json.dumps(s3_policy_document),
        Description='Policy for reading documents from s3')

    kb_policy = iam_client.create_policy(
        PolicyName=kb_policy_name,
        PolicyDocument=json.dumps(bedrock_kb_retrival_policy_document),
        Description='Policy for retrive from kb')

    # create bedrock execution role
    bedrock_kb_execution_role = iam_client.create_role(
        RoleName=bedrock_execution_role_name,
        AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
        Description='Amazon Bedrock Knowledge Base Execution Role for accessing OSS and S3',
        MaxSessionDuration=3600
    )

    # fetch arn of the policies and role created above
    bedrock_kb_execution_role_arn = bedrock_kb_execution_role['Role']['Arn']
    s3_policy_arn = s3_policy["Policy"]["Arn"]
    fm_policy_arn = fm_policy["Policy"]["Arn"]
    kb_policy_arn = kb_policy["Policy"]["Arn"]

    # attach policies to Amazon Bedrock execution role
    iam_client.attach_role_policy(
        RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
        PolicyArn=fm_policy_arn
    )
    iam_client.attach_role_policy(
        RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
        PolicyArn=s3_policy_arn
    )
    iam_client.attach_role_policy(
        RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
        PolicyArn=kb_policy_arn
    )
    
    return bedrock_kb_execution_role


def create_oss_policy_attach_bedrock_execution_role(collection_id, bedrock_kb_execution_role):
    # define oss policy document
    oss_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "aoss:APIAccessAll"
                ],
                "Resource": [
                    f"arn:aws:aoss:{region_name}:{account_number}:collection/{collection_id}"
                ]
            }
        ]
    }
    oss_policy = iam_client.create_policy(
        PolicyName=oss_policy_name,
        PolicyDocument=json.dumps(oss_policy_document),
        Description='Policy for accessing opensearch serverless',
    )
    oss_policy_arn = oss_policy["Policy"]["Arn"]
    print("Opensearch serverless arn: ", oss_policy_arn)

    iam_client.attach_role_policy(
        RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
        PolicyArn=oss_policy_arn
    )
    return None


def create_policies_in_oss(vector_store_name, aoss_client, bedrock_kb_execution_role_arn):
    encryption_policy = aoss_client.create_security_policy(
        name=encryption_policy_name,
        policy=json.dumps(
            {
                'Rules': [{'Resource': ['collection/' + vector_store_name],
                           'ResourceType': 'collection'}],
                'AWSOwnedKey': True
            }),
        type='encryption'
    )

    network_policy = aoss_client.create_security_policy(
        name=network_policy_name,
        policy=json.dumps(
            [
                {'Rules': [{'Resource': ['collection/' + vector_store_name],
                            'ResourceType': 'collection'}],
                 'AllowFromPublic': True}
            ]),
        type='network'
    )
    access_policy = aoss_client.create_access_policy(
        name=access_policy_name,
        policy=json.dumps(
            [
                {
                    'Rules': [
                        {
                            'Resource': ['collection/' + vector_store_name],
                            'Permission': [
                                'aoss:CreateCollectionItems',
                                'aoss:DeleteCollectionItems',
                                'aoss:UpdateCollectionItems',
                                'aoss:DescribeCollectionItems'],
                            'ResourceType': 'collection'
                        },
                        {
                            'Resource': ['index/' + vector_store_name + '/*'],
                            'Permission': [
                                'aoss:CreateIndex',
                                'aoss:DeleteIndex',
                                'aoss:UpdateIndex',
                                'aoss:DescribeIndex',
                                'aoss:ReadDocument',
                                'aoss:WriteDocument'],
                            'ResourceType': 'index'
                        }],
                    'Principal': [identity, bedrock_kb_execution_role_arn],
                    'Description': 'Easy data policy'}
            ]),
        type='data'
    )
    return encryption_policy, network_policy, access_policy

# create lambda role
def create_lambda_role(agent_name):
    # Create IAM Role for the Lambda function
    try:
        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "lambda.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }

        assume_role_policy_document_json = json.dumps(assume_role_policy_document)

        lambda_iam_role = iam_client.create_role(
            RoleName=lambda_function_role_name,
            AssumeRolePolicyDocument=assume_role_policy_document_json
        )

        # Pause to make sure role is created
        time.sleep(10)
    except iam_client.exceptions.EntityAlreadyExistsException:
        lambda_iam_role = iam_client.get_role(RoleName=lambda_function_role_name)

    # Attach the AWSLambdaBasicExecutionRole policy
    iam_client.attach_role_policy(
        RoleName=lambda_function_role_name,
        PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
    )

    # Bedrock FM invoke policy
    foundation_model_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                ],
                "Resource": "*"
            }
        ]
    }
    
    # S3 policy for writing PlantUML diagrams to SageMaker default bucket
    sagemaker_s3_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:PutObject",
                    "s3:GetObject",
                    "s3:ListBucket"
                ],
                "Resource": [
                    # SageMaker default bucket pattern: sagemaker-{region}-{account_id}
                    f"arn:aws:s3:::sagemaker-{region_name}-{account_number}",
                    f"arn:aws:s3:::sagemaker-{region_name}-{account_number}/*"
                ]
            }
        ]
    }
    
    # Create policies based on the policy documents
    fm_policy = iam_client.create_policy(
        PolicyName=lambda_fm_policy_name,
        PolicyDocument=json.dumps(foundation_model_policy_document),
        Description='Policy for accessing foundation model',
    )

    # Create S3 policy for PlantUML diagram storage
    sagemaker_s3_policy_name = f'AWSLambdaSageMakerS3Policy_{suffix}'
    s3_policy = iam_client.create_policy(
        PolicyName=sagemaker_s3_policy_name,
        PolicyDocument=json.dumps(sagemaker_s3_policy_document),
        Description='Policy for writing PlantUML diagrams to SageMaker default S3 bucket',
    )

    # Attach the policies to the Lambda function's role
    iam_client.attach_role_policy(
        RoleName=lambda_function_role_name,
        PolicyArn=fm_policy['Policy']['Arn']
    )
    
    iam_client.attach_role_policy(
        RoleName=lambda_function_role_name,
        PolicyArn=s3_policy['Policy']['Arn']
    )

    return lambda_iam_role

def create_lambda(lambda_file, lambda_function_name, lambda_iam_role):
    # add to function

    # Package up the lambda function code
    s = BytesIO()
    z = zipfile.ZipFile(s, 'w')
    z.write(lambda_file)
    z.close()
    zip_content = s.getvalue()

    # Create Lambda Function
    lambda_function = lambda_client.create_function(
        FunctionName=lambda_function_name,
        Runtime='python3.12',
        Timeout=60,
        Role=lambda_iam_role['Role']['Arn'],
        Code={'ZipFile': zip_content},
        Handler='lambda_function.lambda_handler'
    )
    return lambda_function
    

def delete_iam_role_and_policies():
    fm_policy_arn = f"arn:aws:iam::{account_number}:policy/{fm_policy_name}"
    s3_policy_arn = f"arn:aws:iam::{account_number}:policy/{s3_policy_name}"
    oss_policy_arn = f"arn:aws:iam::{account_number}:policy/{oss_policy_name}"
    kb_policy_arn = f"arn:aws:iam::{account_number}:policy/{kb_policy_name}"
    lambda_fn_policy_arn = f"arn:aws:iam::{account_number}:policy/{lambda_fm_policy_name}"
    sagemaker_s3_policy_arn = f"arn:aws:iam::{account_number}:policy/AWSLambdaSageMakerS3Policy_{suffix}"

    iam_client.detach_role_policy(
        RoleName=bedrock_execution_role_name,
        PolicyArn=s3_policy_arn
    )
    iam_client.detach_role_policy(
        RoleName=bedrock_execution_role_name,
        PolicyArn=fm_policy_arn
    )
    iam_client.detach_role_policy(
        RoleName=bedrock_execution_role_name,
        PolicyArn=kb_policy_arn
    )
    
    iam_client.detach_role_policy(
        RoleName=bedrock_execution_role_name,
        PolicyArn=oss_policy_arn
    )

    iam_client.detach_role_policy(
        RoleName=lambda_function_role_name,
        PolicyArn=lambda_fn_policy_arn
    )

    iam_client.detach_role_policy(
        RoleName=lambda_function_role_name,
        PolicyArn=sagemaker_s3_policy_arn
    )

    iam_client.delete_role(RoleName=bedrock_execution_role_name)
    iam_client.delete_role(RoleName=lambda_function_role_name)
    iam_client.delete_policy(PolicyArn=s3_policy_arn)
    iam_client.delete_policy(PolicyArn=kb_policy_arn)
    iam_client.delete_policy(PolicyArn=fm_policy_arn)
    iam_client.delete_policy(PolicyArn=oss_policy_arn)
    iam_client.delete_policy(PolicyArn=lambda_fn_policy_arn)
    iam_client.delete_policy(PolicyArn=sagemaker_s3_policy_arn)
    return 0


def interactive_sleep(seconds: int):
    dots = ''
    for i in range(seconds):
        dots += '.'
        print(dots, end='\r')
        time.sleep(1)
    print('Done!')


def create_bedrock_execution_role_multi_ds(bucket_names):
    foundation_model_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                ],
                "Resource": "*"
            }
        ]
    }

    s3_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:ListBucket"
                ],
                "Resource": [item for sublist in [[f'arn:aws:s3:::{bucket}', f'arn:aws:s3:::{bucket}/*'] for bucket in bucket_names] for item in sublist], 
                "Condition": {
                    "StringEquals": {
                        "aws:ResourceAccount": f"{account_number}"
                    }
                }
            }
        ]
    }

    assume_role_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "bedrock.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    # create policies based on the policy documents
    fm_policy = iam_client.create_policy(
        PolicyName=fm_policy_name,
        PolicyDocument=json.dumps(foundation_model_policy_document),
        Description='Policy for accessing foundation model',
    )

    s3_policy = iam_client.create_policy(
        PolicyName=s3_policy_name,
        PolicyDocument=json.dumps(s3_policy_document),
        Description='Policy for reading documents from s3')

    # create bedrock execution role
    bedrock_kb_execution_role = iam_client.create_role(
        RoleName=bedrock_execution_role_name,
        AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
        Description='Amazon Bedrock Knowledge Base Execution Role for accessing OSS and S3',
        MaxSessionDuration=3600
    )

    # fetch arn of the policies and role created above
    bedrock_kb_execution_role_arn = bedrock_kb_execution_role['Role']['Arn']
    s3_policy_arn = s3_policy["Policy"]["Arn"]
    fm_policy_arn = fm_policy["Policy"]["Arn"]

    # attach policies to Amazon Bedrock execution role
    iam_client.attach_role_policy(
        RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
        PolicyArn=fm_policy_arn
    )
    iam_client.attach_role_policy(
        RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
        PolicyArn=s3_policy_arn
    )
    return bedrock_kb_execution_role

# Converse API invoke model
def invoke_bedrock_model(client, id, prompt, max_tokens=2000, temperature=0, top_p=0.9):
    response = ""
    try:
        response = client.converse(
            modelId=id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            inferenceConfig={
                "temperature": temperature,
                "maxTokens": max_tokens,
                "topP": top_p
            }
            #additionalModelRequestFields={
            #}
        )
    except Exception as e:
        print(e)
        result = "Model invocation error"
    try:
        result = response['output']['message']['content'][0]['text'] \
        + '\n--- Latency: ' + str(response['metrics']['latencyMs']) \
        + 'ms - Input tokens:' + str(response['usage']['inputTokens']) \
        + ' - Output tokens:' + str(response['usage']['outputTokens']) + ' ---\n'
        return result
    except Exception as e:
        print(e)
        result = "Output parsing error"
    return result


# Converse API streaming
def invoke_bedrock_model_stream(client, id, prompt, 
                                max_tokens=2000, temperature=0, top_p=0.9):
    response = ""
    response = client.converse_stream(
        modelId=id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        inferenceConfig={
            "temperature": temperature,
            "maxTokens": max_tokens,
            "topP": top_p
        }
    )
    # Extract and print the response text in real-time.
    for event in response['stream']:
        if 'contentBlockDelta' in event:
            chunk = event['contentBlockDelta']
            sys.stdout.write(chunk['delta']['text'])
            sys.stdout.flush()
    return