# Lambda Setup

Render pauses deployments, so we gotta do this using Lambda and EventBridge on AWS

## Setup

### 1. Lambda Layer

1. Run these commands to create the lamdba layers for the libraries:

    ```sh
    docker build -t psycopg2-layer-py313-x86 .
    docker create --name temp psycopg2-layer-py313-x86
    docker cp temp:/psycopg2-py313-layer.zip .
    docker rm temp
    ```

2. Upload the layer

    ```sh
    aws lambda publish-layer-version \
      --layer-name psycopg2-py313-x86-layer \
      --zip-file fileb://psycopg2-py313-layer.zip \
      --compatible-runtimes python3.13 \
      --description "psycopg2-binary for Python 3.13, x86_64 architecture"
    ```

3. Update the lambda function

    ```sh
    aws lambda update-function-configuration \
      --function-name YOUR_FUNCTION_NAME \
      --layers arn:aws:lambda:<region>:<account-id>:layer:psycopg2-py313-x86-layer:<version>
    ```

### 2. Setup CRON triggers for the Lambda Function

1. Use EventBridge triggers.

    NOTE: The CRON schedule is in UTC, so we need to translate to SG timing
    - i.e. 5:30am to 1:00 am -> 2130 - 1559
    - /10 21-23 * * ? *
    - /10 0-15 * * ? *
