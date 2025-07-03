"""
Lambda Handler for VIGIA Medical AI FastAPI Application
======================================================

AWS Lambda container handler using Mangum for ASGI compatibility.
Wraps the FastAPI application for serverless deployment.
"""

import os
import logging
from mangum import Mangum
from .main import app

# Configure logging for Lambda
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Lambda deployment flag
os.environ["LAMBDA_DEPLOYMENT"] = "true"

# Create the Mangum handler for Lambda
handler = Mangum(
    app,
    lifespan="off",  # Disable lifespan for Lambda compatibility
    api_gateway_base_path=None,  # Auto-detect base path
    text_mime_types=[
        "application/json",
        "application/javascript",
        "application/xml",
        "application/vnd.api+json",
        "text/css",
        "text/html",
        "text/plain",
        "text/xml"
    ]
)

# For debugging in Lambda
def lambda_handler(event, context):
    """
    AWS Lambda entry point
    
    Args:
        event: API Gateway event
        context: Lambda context
        
    Returns:
        API Gateway response
    """
    logger.info(f"Lambda event: {event}")
    logger.info(f"Lambda context: {context}")
    
    try:
        response = handler(event, context)
        logger.info(f"Lambda response: {response}")
        return response
    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": f'{{"error": "Internal server error: {str(e)}"}}'
        }

# Export both handlers for flexibility
__all__ = ["handler", "lambda_handler"]