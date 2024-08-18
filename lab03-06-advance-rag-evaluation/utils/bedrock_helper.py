import json
from dataclasses import dataclass, field
import os
import boto3
import time
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

def create_index(host, region, credentials, index_name, embedding_dim):
    service = 'aoss'
    auth = AWSV4SignerAuth(credentials, region, service)
    aoss_pyclient = OpenSearch(
        hosts = [{'host': host, 'port': 443}],
        http_auth = auth,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection,
        pool_maxsize = 20
    )

    index_body = {
        "settings": {
            "index": {
                "knn": "true",
                "number_of_shards": "2",
                "knn.algo_param": {
                  "ef_search": "512"
                },
            }
        },
        "mappings": {
          "properties": {
            "AMAZON_BEDROCK_METADATA": {
              "type": "text",
              "index": "false"
            },
            "AMAZON_BEDROCK_TEXT_CHUNK": {
              "type": "text"
            },
            "bedrock-knowledge-base-default-vector": {
              "type": "knn_vector",
              "dimension": embedding_dim,
              "method": {
                "engine": "faiss",
                "name": "hnsw",
                "parameters": {}
              }
            }
          }
        }
    }
    response = aoss_pyclient.indices.create(index_name, body=index_body)
    return response

def _create_knowledge_base(knowledge_base_name, role_arn, embedding_model_arn, collection_arn, index_name, bucket, s3_prefix):
    bedrock_agent = boto3.client("bedrock-agent")
    response = bedrock_agent.create_knowledge_base(
        name=knowledge_base_name,
        description='Knowledge Base for Bedrock',
        roleArn=role_arn,
        knowledgeBaseConfiguration={
            'type': 'VECTOR',
            'vectorKnowledgeBaseConfiguration': {
                'embeddingModelArn': embedding_model_arn
            }
        },
        storageConfiguration={
            'type': 'OPENSEARCH_SERVERLESS',
            'opensearchServerlessConfiguration': {
                'collectionArn': collection_arn,
                'vectorIndexName': index_name,
                'fieldMapping': {
                    'vectorField':  "bedrock-knowledge-base-default-vector",
                    'textField': 'AMAZON_BEDROCK_TEXT_CHUNK',
                    'metadataField': 'AMAZON_BEDROCK_METADATA'
                }
            }
        }
    )
    knowledge_base_id = response['knowledgeBase']['knowledgeBaseId']
    knowledge_base_name = response['knowledgeBase']['name']

    response = bedrock_agent.create_data_source(
    knowledgeBaseId=knowledge_base_id,
    name=f"{knowledge_base_name}-ds",
    dataSourceConfiguration={
        'type': 'S3',
        's3Configuration': {
            'bucketArn': f"arn:aws:s3:::{bucket}",
            'inclusionPrefixes': [
                f"{s3_prefix}/",
            ]
        }
    },
    vectorIngestionConfiguration={
            'chunkingConfiguration': {
                'chunkingStrategy': 'FIXED_SIZE',
                'fixedSizeChunkingConfiguration': {
                    'maxTokens': 300,
                    'overlapPercentage': 10
                }
            }
        }
    )
    data_source_id = response['dataSource']['dataSourceId']

    # Check status to make sure the KB is created before we could start the ingestion job.
    kb_status = bedrock_agent.get_knowledge_base(knowledgeBaseId=knowledge_base_id)
    while kb_status not in [ "ACTIVE", "FAILED", "DELETE_UNSUCCESSFUL" ]:
        response = bedrock_agent.get_knowledge_base(knowledgeBaseId=knowledge_base_id)
        kb_status = response['knowledgeBase']['status']

    if kb_status != "ACTIVE":
        raise Exception("Bedrock Knowledgebase did not create successfully. Please check for error before proceeding") 

    print(f"Bedrock KnowledgeBase {knowledge_base_id} created successfully")
    response = bedrock_agent.start_ingestion_job(
        knowledgeBaseId=knowledge_base_id,
        dataSourceId=data_source_id,
    )

    ingestion_job_id = response['ingestionJob']['ingestionJobId']
    ingestion_job_status = response['ingestionJob']['status']

    while ingestion_job_status not in ['COMPLETE', 'FAILED']:
        response = bedrock_agent.get_ingestion_job(
            knowledgeBaseId=knowledge_base_id,
            dataSourceId=data_source_id,
            ingestionJobId=ingestion_job_id
        )
        ingestion_job_status = response['ingestionJob']['status']
        time.sleep(5)
    return knowledge_base_id


def create_knowledge_base(knowledge_base_name, bedrock_kb_execution_role_arn, embedding_model_arn, embedding_dim, s3_bucket, s3_prefix, oss_host, oss_collection_id, oss_collection_arn, index_name, region, credentials):
    create_index(oss_host, region, credentials, index_name, embedding_dim)
    time.sleep(60) # sleeps until the IAM role is created successfully
    knowledge_base_id = _create_knowledge_base(knowledge_base_name, bedrock_kb_execution_role_arn, embedding_model_arn, oss_collection_arn, index_name, s3_bucket, s3_prefix)
    return knowledge_base_id