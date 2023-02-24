import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

s3 = boto3.resource('s3',
                    endpoint_url='https://c5496cc41ca6d42c8358101ad551f1b4.r2.cloudflarestorage.com',
                    aws_access_key_id='4e57071dba2f0fb9f5a2af8037e95e82',
                    aws_secret_access_key='0bdd57295ce8d381cb6a9ab486a0f02abbd9ab9eff8a7f521bea7cd03de8189c',
                    config=Config(signature_version='s3v4')
                    )

url = s3.meta.client.generate_presigned_url('get_object', Params={
    'Bucket': 'cms', 'Key': 'b756ec346cadf3eb71c98250/2f73eadd.png'}, ExpiresIn=3600)


print(url)
