import boto3

# Initialize a Boto3 client for SQS
sqs = boto3.client('sqs', region_name='ap-southeast-1')

# Specify your SQS queue URL
queue_url = 'https://sqs.ap-southeast-1.amazonaws.com/467469515596/xtts'

try:
    # Receive a message from the SQS queue
    response = sqs.receive_message(
        QueueUrl=queue_url,
        AttributeNames=[
            'SentTimestamp'
        ],
        MaxNumberOfMessages=1,
        MessageAttributeNames=[
            'All'
        ],
        VisibilityTimeout=30,  # The duration (in seconds) that the received messages are hidden from subsequent retrieve requests after being retrieved by a receive message request
        WaitTimeSeconds=0  # The duration (in seconds) for which the call waits for a message to arrive in the queue before returning
    )

    if 'Messages' in response:
        message = response['Messages'][0]
        receipt_handle = message['ReceiptHandle']

        print("Message Received: ")
        print(message['Body'])

        # Delete received message from queue to prevent it from being processed again
        sqs.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle
        )
        print("Message deleted from the queue.")
    else:
        print("No messages to process.")
except Exception as e:
    print(f"An error occurred: {e}")
