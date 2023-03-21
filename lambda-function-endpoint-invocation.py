## Lambda function for asynchronous invocation of endpoint

import json
import urllib
from predictor import preprocess

def lambda_handler(event, context):
    #for r in event['query']['Records']
    for r in event['Records']:
        bucket = r['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(r['s3']['object']['key'], encoding='utf-8')
        uri = "/".join([bucket, key])
        output = preprocess(uri)
    
    return {
        'statusCode': 200,
        'body': {
            
            "s3_bucket": bucket,
            "s3_key": uri ,
            "predicted_category": output
                }
            }

# Predictor function .py file to invoke endpoint
import boto3
import os
import io
import boto3
import json
import base64
ENDPOINT_NAME = 'pytorch-inference-2023-02-10-05-41-51-640'

runtime= boto3.client('runtime.sagemaker')		
def preprocess(s3_input_uri):
    
    with open(s3_input_uri, "rb") as f:
        image = base64.b64encode(f.read())
		# image = base64.b64encode(image_data)
    	# Make a prediction:
        response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='image/png',
                                       Body=image)
        predicted_category = np.argmax(response,1) + 1 
        return predicted_category