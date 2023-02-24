import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

s3 = boto3.resource('s3',
                    endpoint_url='https://xxxxxxxxxxxxxxxxxxx.r2.cloudflarestorage.com',
                    aws_access_key_id='xxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
                    aws_secret_access_key='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
                    config=Config(signature_version='s3v4')
                    )

url = s3.meta.client.generate_presigned_url('get_object', Params={
    'Bucket': 'cms', 'Key': 'b756ec346cadf3eb71c98250/2f73eadd.png'}, ExpiresIn=3600)


print(url)
