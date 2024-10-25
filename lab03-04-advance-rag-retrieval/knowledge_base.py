import json
import boto3
import time
from botocore.exceptions import ClientError
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, RequestError
import pprint
from retrying import retry
import zipfile
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

valid_embedding_models = ["cohere.embed-multilingual-v3", 
                          "cohere.embed-english-v3", 
                          "amazon.titan-embed-text-v1", 
                          "amazon.titan-embed-text-v2:0"]

# create a dictionary with model id as key and context length as value
embedding_context_dimensions = {
    "cohere.embed-multilingual-v3": 512,
    "cohere.embed-english-v3": 512,
    "amazon.titan-embed-text-v1": 1536,
    "amazon.titan-embed-text-v2:0": 1024
}
pp = pprint.PrettyPrinter(indent=2)


def interactive_sleep(seconds: int):
    """
    Support functionality to induce an artificial 'sleep' to the code in order to wait for resources to be available
    Args:
        seconds (int): number of seconds to sleep for
    """
    dots = ''
    for i in range(seconds):
        dots += '.'
        print(dots, end='\r')
        time.sleep(1)


class BedrockKnowledgeBase:

    def __init__(
            self,
            kb_name,
            kb_description,
            data_bucket_name,
            data_prefix,
            vector_collection_arn,
            vector_collection_id,
            vector_host,
            bedrock_kb_execution_role_arn,
            index_name,
            suffix,
            embedding_model="amazon.titan-embed-text-v2:0",
            chunking_strategy="FIXED_SIZE"
    ):
        """
        Class initializer
        Args:
            kb_name(str): The name of the Knowledge Base.
            kb_description(str): The description of the Knowledge Base.
            data_bucket_name(str): The name of the S3 bucket to be used as the data source for the Knowledge Base.
            lambda_function_name(str): The name of the Lambda function to be used for custom chunking strategy.
            embedding_model(str): The embedding model to be used for the Knowledge Base.
            chunking_strategy(str): The chunking strategy to be used for the Knowledge Base.
            suffix(str): A suffix to be used for naming resources.
        """
        self.boto3_session = boto3.session.Session()
        self.region_name = self.boto3_session.region_name
        self.sts_client = self.boto3_session.client('sts')
        self.account_id = self.sts_client.get_caller_identity()["Account"]
        self.credentials = self.boto3_session.get_credentials()

        # oss attributes
        self.awsauth = AWSV4SignerAuth(self.credentials, self.region_name, 'aoss')
        self.aoss_client = self.boto3_session.client('opensearchserverless')
        self.bucket_name = data_bucket_name
        self.data_prefix = data_prefix
        self.host = vector_host
        self.collection_arn = vector_collection_arn
        self.collection_id = vector_collection_id
        self.index_name = index_name
        
        
        self.oss_client = OpenSearch(
            hosts=[{'host': self.host, 'port': 443}],
            http_auth=self.awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )

        # knowledge base attributes
        self.kb_name = kb_name
        self.kb_description = kb_description
        self.chunking_strategy = chunking_strategy
        if embedding_model not in valid_embedding_models:
            valid_embeddings_str = str(valid_embedding_models)
            raise ValueError(f"Invalid embedding model. Your embedding model should be one of {valid_embeddings_str}")

        # bedrock attributes
        self.s3_client = self.boto3_session.client('s3')
        self.bedrock_agent_client = self.boto3_session.client('bedrock-agent')
        self.embedding_model = embedding_model
        self.kb_execution_role_name = bedrock_kb_execution_role_arn

        print("========================================================================================")
        print(f"Step 1 - Creating OSS Vector Index")
        self.create_vector_index()

        print("========================================================================================")
        print(f"Step 2 - Creating Knowledge Base")
        self.knowledge_base, self.data_source = self.create_knowledge_base()
        print("========================================================================================")
 
    def create_vector_index(self):
        """
        Create OpenSearch Serverless vector index. If existent, ignore
        """
        body_json = {
            "settings": {
                "index.knn": "true",
                "number_of_shards": 1,
                "knn.algo_param.ef_search": 512,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": embedding_context_dimensions[self.embedding_model], # use dimension as per the context length of embeddings model selected.
                        "method": {
                            "name": "hnsw",
                            "engine": "faiss",
                            "space_type": "l2"
                        },
                    },
                    "text": {
                        "type": "text"
                    },
                    "text-metadata": {
                        "type": "text"}
                }
            }
        }

        # Create index
        try:
            response = self.oss_client.indices.create(index=self.index_name, body=json.dumps(body_json))
            print('\nCreating index:')
            pp.pprint(response)

            # index creation can take up to a minute
            interactive_sleep(60)
        except RequestError as e:
            # you can delete the index if its already exists
            # oss_client.indices.delete(index=index_name)
            print(
                f'Error while trying to create the index, with error {e.error}\nyou may unmark the delete above to '
                f'delete, and recreate the index')
    
    def create_chunking_strategy_config(self, strategy):
        configs = {
            "NONE": {
                "chunkingConfiguration": {"chunkingStrategy": "NONE"}
            },
            "FIXED_SIZE": {
                "chunkingConfiguration": {
                "chunkingStrategy": "FIXED_SIZE",
                "fixedSizeChunkingConfiguration": {
                    "maxTokens": 300,
                    "overlapPercentage": 20
                    }
                }
            },
            "HIERARCHICAL": {
                "chunkingConfiguration": {
                "chunkingStrategy": "HIERARCHICAL",
                "hierarchicalChunkingConfiguration": {
                    "levelConfigurations": [{"maxTokens": 1500}, {"maxTokens": 300}],
                    "overlapTokens": 60
                    }
                }
            },
            "SEMANTIC": {
                "chunkingConfiguration": {
                "chunkingStrategy": "SEMANTIC",
                "semanticChunkingConfiguration": {
                    "maxTokens": 300,
                    "bufferSize": 1,
                    "breakpointPercentileThreshold": 95}
                }
            }
        }
        return configs.get(strategy, configs["NONE"])

    @retry(wait_random_min=1000, wait_random_max=2000, stop_max_attempt_number=7)
    def create_knowledge_base(self):
        """
        Create Knowledge Base and its Data Source. If existent, retrieve
        """
        opensearch_serverless_configuration = {
            "collectionArn": self.collection_arn,
            "vectorIndexName": self.index_name,
            "fieldMapping": {
                "vectorField": "vector",
                "textField": "text",
                "metadataField": "text-metadata"
            }
        }

        chunking_strategy_configuration = {}
        # vectorIngestionConfiguration = {}

        print(f"Creating KB with chunking strategy - {self.chunking_strategy}")
        chunking_strategy_configuration = self.create_chunking_strategy_config(self.chunking_strategy)
        print("============Chunking config========\n", chunking_strategy_configuration)

        # The data source to ingest documents from, into the OpenSearch serverless knowledge base index
        s3_configuration = {
            "bucketArn": f"arn:aws:s3:::{self.bucket_name}",
            "inclusionPrefixes": self.data_prefix # you can use this if you want to create a KB using data within s3 prefixes.
        }

        # The embedding model used by Bedrock to embed ingested documents, and realtime prompts
        embedding_model_arn = f"arn:aws:bedrock:{self.region_name}::foundation-model/{self.embedding_model}"
        try:
            create_kb_response = self.bedrock_agent_client.create_knowledge_base(
                name=self.kb_name,
                description=self.kb_description,
                roleArn=self.kb_execution_role_name,
                knowledgeBaseConfiguration={
                    "type": "VECTOR",
                    "vectorKnowledgeBaseConfiguration": {
                        "embeddingModelArn": embedding_model_arn
                    }
                },
                storageConfiguration={
                    "type": "OPENSEARCH_SERVERLESS",
                    "opensearchServerlessConfiguration": opensearch_serverless_configuration
                }
            )
            kb = create_kb_response["knowledgeBase"]
            pp.pprint(kb)
        except self.bedrock_agent_client.exceptions.ConflictException:
            kbs = self.bedrock_agent_client.list_knowledge_bases(
                maxResults=100
            )
            kb_id = None
            for kb in kbs['knowledgeBaseSummaries']:
                if kb['name'] == self.kb_name:
                    kb_id = kb['knowledgeBaseId']
            response = self.bedrock_agent_client.get_knowledge_base(knowledgeBaseId=kb_id)
            kb = response['knowledgeBase']
            pp.pprint(kb)

        # Create a DataSource in KnowledgeBase
        try:
            print(self.kb_name)
            print(kb['knowledgeBaseId'])
            print(s3_configuration)
            create_ds_response = self.bedrock_agent_client.create_data_source(
                name=self.kb_name,
                description=self.kb_description,
                knowledgeBaseId=kb['knowledgeBaseId'],
                dataSourceConfiguration={
                    "type": "S3",
                    "s3Configuration": s3_configuration
                },
                vectorIngestionConfiguration = chunking_strategy_configuration, 
                dataDeletionPolicy='RETAIN'
            )
            ds = create_ds_response["dataSource"]
            pp.pprint(ds)
        except self.bedrock_agent_client.exceptions.ConflictException:
            ds_id = self.bedrock_agent_client.list_data_sources(
                knowledgeBaseId=kb['knowledgeBaseId'],
                maxResults=100
            )['dataSourceSummaries'][0]['dataSourceId']
            get_ds_response = self.bedrock_agent_client.get_data_source(
                dataSourceId=ds_id,
                knowledgeBaseId=kb['knowledgeBaseId']
            )
            ds = get_ds_response["dataSource"]
            pp.pprint(ds)
        return kb, ds

    def start_ingestion_job(self):
        """
        Start an ingestion job to synchronize data from an S3 bucket to the Knowledge Base
        """
        # Start an ingestion job
        start_job_response = self.bedrock_agent_client.start_ingestion_job(
            knowledgeBaseId=self.knowledge_base['knowledgeBaseId'],
            dataSourceId=self.data_source["dataSourceId"]
        )
        job = start_job_response["ingestionJob"]
        pp.pprint(job)
        # Get job
        while job['status'] != 'COMPLETE':
            get_job_response = self.bedrock_agent_client.get_ingestion_job(
                knowledgeBaseId=self.knowledge_base['knowledgeBaseId'],
                dataSourceId=self.data_source["dataSourceId"],
                ingestionJobId=job["ingestionJobId"]
            )
            job = get_job_response["ingestionJob"]
        pp.pprint(job)
        interactive_sleep(40)

    def get_knowledge_base_id(self):
        """
        Get Knowledge Base Id
        """
        pp.pprint(self.knowledge_base["knowledgeBaseId"])
        return self.knowledge_base["knowledgeBaseId"]

    def get_bucket_name(self):
        """
        Get the name of the bucket connected with the Knowledge Base Data Source
        """
        pp.pprint(f"Bucket connected with KB: {self.bucket_name}")
        return self.bucket_name

    def delete_kb(self, delete_s3_bucket=False, delete_iam_roles_and_policies=True, delete_lambda_function=False):
        """
        Delete the Knowledge Base resources
        Args:
            delete_s3_bucket (bool): boolean to indicate if s3 bucket should also be deleted
            delete_iam_roles_and_policies (bool): boolean to indicate if IAM roles and Policies should also be deleted
            delete_lambda_function (bool): boolean to indicate if Lambda function should also be deleted
        """
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # delete vector index and collection from vector store
            try:
                self.aoss_client.delete_collection(id=self.collection_id)
                self.aoss_client.delete_access_policy(
                    type="data",
                    name=self.access_policy_name
                )
                self.aoss_client.delete_security_policy(
                    type="network",
                    name=self.network_policy_name
                )
                self.aoss_client.delete_security_policy(
                    type="encryption",
                    name=self.encryption_policy_name
                )
                print("======== Vector Index, collection and associated policies deleted =========")
            except Exception as e:
                print(e)

            # delete knowledge base and vector store.
            
            try:
                self.bedrock_agent_client.delete_data_source(
                    dataSourceId=self.data_source["dataSourceId"],
                    knowledgeBaseId=self.knowledge_base['knowledgeBaseId']
                )
                self.bedrock_agent_client.delete_knowledge_base(
                    knowledgeBaseId=self.knowledge_base['knowledgeBaseId']
                )
                print("======== Knowledge base and data source deleted =========")
            except self.bedrock_agent_client.exceptions.ResourceNotFoundException as e:
                print("Resource not found", e)
                pass
            except Exception as e:
                print(e)

            # delete s3 bucket
            if delete_s3_bucket==True:
                    self.delete_s3()
                    
            # delete IAM role and policies
            if delete_iam_roles_and_policies:
                self.delete_iam_roles_and_policies()
            
            if delete_lambda_function:
                try:
                    self.delete_lambda_function()
                    print(f"Deleted Lambda function {self.lambda_function_name}")
                except self.lambda_client.exceptions.ResourceNotFoundException:
                    print(f"Lambda function {self.lambda_function_name} not found.")