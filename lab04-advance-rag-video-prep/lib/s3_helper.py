import os
import boto3
import sagemaker

def upload_object(bucket, prefix, file):
    
    key = os.path.join(prefix, file)

    s3_client = boto3.client('s3')

    with open(file, "rb") as f:
        response = s3_client.put_object(
            Body=f,
            Bucket=bucket,
            Key=key,
        )
    return response