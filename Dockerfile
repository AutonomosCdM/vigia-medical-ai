FROM public.ecr.aws/lambda/python:3.11

# Copy requirements first for better caching
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code to Lambda task root
COPY . ${LAMBDA_TASK_ROOT}/

# Set environment variables for Lambda
ENV AWS_DEPLOYMENT=true
ENV LAMBDA_DEPLOYMENT=true
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}

# Lambda runtime will call this handler
CMD ["src.web.lambda_handler.handler"]