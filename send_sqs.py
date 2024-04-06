import boto3

# Initialize a Boto3 client for SQS
sqs = boto3.client('sqs', region_name='ap-southeast-1')

# Specify your SQS queue URL
queue_url = 'https://sqs.ap-southeast-1.amazonaws.com/467469515596/xtts'

# The message you want to send
message_body = 'Hello, this is a test message.'

try:
    # Send the message to the SQS queue
    response = sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=message_body
    )
    print(f"Message sent. Message ID: {response['MessageId']}")
except Exception as e:
    print(f"An error occurred: {e}")
