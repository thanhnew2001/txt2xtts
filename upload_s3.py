import boto3
from botocore.exceptions import NoCredentialsError

def upload_file_to_s3(file_name, bucket_name, object_name=None):
    if object_name is None:
        object_name = file_name
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_name, bucket_name, object_name)
        # Generate a presigned URL for the uploaded file
        presigned_url = s3_client.generate_presigned_url('get_object',
                                                         Params={'Bucket': bucket_name,
                                                                 'Key': object_name},
                                                         ExpiresIn=3600) # URL expires in 1 hour
        return presigned_url
    except NoCredentialsError:
        print("Credentials not available")
        return None

# Usage example
file_name = 'female_voice.wav'
bucket_name = 'arn:aws:s3:::xtts'
presigned_url = upload_file_to_s3(file_name, bucket_name)
print(presigned_url)
