# Use the Lambda Python 3.13 base image
FROM --platform=linux/amd64 amazon/aws-lambda-python:3.13

RUN dnf install -y zip

# Install psycopg2-binary into the /opt/python directory
RUN pip install psycopg2-binary requests -t /opt/python

# Package it into a zip file for Lambda Layer
RUN cd /opt && zip -r9 /psycopg2-py313-layer.zip python
