import boto3
import json
import time
from opensearchpy.helpers import bulk
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
# from langchain_community.vectorstores import OpenSearchVectorSearch

class OpenSearchManager:
    def __init__(self, region_name=None):
        self.boto_session = boto3.Session()
        self.region = self.boto_session.region_name if region_name is None else region_name
        self.identity = self.boto_session.client('sts').get_caller_identity()['Arn']
        self.aoss_client = self.boto_session.client('opensearchserverless', 
                                                    region_name=self.region)
        self.service = 'aoss'
        self.credentials = self.boto_session.get_credentials()
        self.auth = AWSV4SignerAuth(self.credentials, self.region, self.service)
        self.client = None
        self.docsearch = None

    def initialize_client(self, host=None):
        if host is None and self.client is None:
            raise ValueError("Host URL is required to initialize the OpenSearch client.")
        elif host is not None:
            self.client = OpenSearch(
                hosts=[{'host': host, 'port': 443}],
                http_auth=self.auth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                pool_maxsize=20
            )
        # If self.client is already set, do nothing

    # def initialize_docsearch(self, host=None, embedding_model=None, index_name=None):
    #     if self.docsearch:
    #         pass
    #     elif host is None:
    #         raise ValueError("Host URL is required to initialize the OpenSearch client.")
    #     elif embedding_model is None:
    #         raise ValueError("Please provide an embedding model.")
    #     elif index_name is None:
    #         raise ValueError("Please provide an embedding model.")
    #     elif host is not None:
    #         self.docsearch = OpenSearchVectorSearch(
    #             embedding_function=embedding_model,
    #             opensearch_url=host,
    #             http_auth=self.auth,
    #             timeout=300,
    #             use_ssl=True,
    #             verify_certs=True,
    #             connection_class=RequestsHttpConnection,
    #             index_name=index_name,
    #             engine="nmslib",
    #         )

    # def langchain_docsearch(self, query, k=5, score=False):
    #     if self.docsearch is None:
    #         raise ValueError("Langchain client is not initialized. Call 'initialize_docsearch' first.")

    #     if score:
    #         results = self.docsearch.similarity_search_with_score(summary, k=k, vector_field="vector_field")
    #     else:
    

    
            
        
        
    
    def bulk_index_ingestion(self, index_name, data):
        if self.client is None:
            raise ValueError("OpenSearch client is not initialized. Call 'initialize_client' first.")

        success, failed = bulk(
            self.client,
            data,
            index=index_name,
            raise_on_exception=True
        )

        print(f"Indexed {success} documents")

        if len(failed) > 0:
            print(f"Failed to index {failed} documents")

        return success, failed

    def create_opensearch_collection(self, vector_store_name="", index_name="index", encryption_policy_name="ep", network_policy_name="np", access_policy_name="ap"):
        print(vector_store_name)
        security_policy = self.aoss_client.create_security_policy(
            name=encryption_policy_name,
            policy=json.dumps(
                {
                    'Rules': [{'Resource': ['collection/' + vector_store_name],
                               'ResourceType': 'collection'}],
                    'AWSOwnedKey': True
                }),
            type='encryption'
        )

        network_policy = self.aoss_client.create_security_policy(
            name=network_policy_name,
            policy=json.dumps(
                [
                    {'Rules': [{'Resource': ['collection/' + vector_store_name],
                                'ResourceType': 'collection'}],
                     'AllowFromPublic': True}
                ]),
            type='network'
        )

        collection = self.aoss_client.create_collection(name=vector_store_name, type='VECTORSEARCH')

        while True:
            status = self.aoss_client.list_collections(collectionFilters={'name': vector_store_name})['collectionSummaries'][0]['status']
            if status in ('ACTIVE', 'FAILED'):
                break
            time.sleep(10)

        access_policy = self.aoss_client.create_access_policy(
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
                        'Principal': [self.identity],
                        'Description': 'Easy data policy'}
                ]),
            type='data'
        )

        host = collection['createCollectionDetail']['id'] + '.' + self.region + '.aoss.amazonaws.com'
        self.initialize_client(host)
        return host

    def create_index(self, index_name="", index_body={}):
        if self.client is None:
            raise ValueError("OpenSearch client is not initialized. Call 'initialize_client' first.")

        try:
            response = self.client.indices.create(index_name, body=index_body)
            print(json.dumps(response, indent=2))
        except Exception as ex:
            print(ex)
        # describe new vector index
        try:
            response = self.client.indices.get(index_name)
        except Exception as ex:
            print(ex)

        return json.dumps(response, indent=2)

    # List available indexes
    def list_indexes(self):
        if self.client is None:
            raise ValueError("OpenSearch client is not initialized. Call 'initialize_client' first.")

        # List all indexes
        index_list = self.client.indices.get_alias().keys()
        return index_list

    # Remove index
    def remove_index(self, index_name):
        if self.client is None:
            raise ValueError("OpenSearch client is not initialized. Call 'initialize_client' first.")
    
        self.client.indices.delete(index=index_name)
        print(f"Deleted index: {index_name}")
    
    # embed and then search
    def opensearch_query(self, query, index_name=""):
        
        if self.client is None:
            raise ValueError("OpenSearch client is not initialized. Call 'initialize_client' first.")
        
        response = self.client.search(body=query, index=index_name)
    
        return response["hits"]["hits"]