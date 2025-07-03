#!/usr/bin/env python3
"""
VIGIA Medical AI - Complete AWS CDK Stack
9-Agent Medical AI System for Pressure Injury Detection with HIPAA Compliance

Architecture:
- Bedrock: MedGemma/Claude models for medical analysis
- Lambda: Serverless 9-agent medical processing
- Step Functions: Agent orchestration workflow
- DynamoDB: Medical state and audit trail management
- S3: Medical imaging and PHI-protected storage
- API Gateway: HIPAA-compliant medical interfaces
"""

from aws_cdk import (
    Stack,
    Duration,
    aws_lambda as _lambda,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as sfn_tasks,
    aws_dynamodb as dynamodb,
    aws_s3 as s3,
    aws_iam as iam,
    aws_apigateway as apigateway,
    aws_events as events,
    aws_events_targets as targets,
    aws_ecr as ecr,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cw_actions,
    aws_sns as sns,
    aws_logs as logs,
    aws_route53 as route53,
    aws_route53_targets as route53_targets,
    aws_certificatemanager as acm,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    RemovalPolicy,
    CfnOutput
)
from constructs import Construct

class VigiaStack(Stack):
    """Complete VIGIA Medical AI Stack"""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # =============================================================================
        # STORAGE LAYER - MEDICAL DATA & PHI COMPLIANCE
        # =============================================================================
        
        # S3 Bucket for medical imaging and PHI-protected documents
        self.medical_storage = s3.Bucket(
            self, "VigiamedicalStorage",
            bucket_name="vigia-medical-storage",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="ArchiveMedicalData",
                    noncurrent_version_transitions=[
                        s3.NoncurrentVersionTransition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(90)
                        )
                    ]
                )
            ],
            removal_policy=RemovalPolicy.DESTROY
        )

        # DynamoDB Tables for 9-agent medical coordination
        self.agents_state_table = dynamodb.Table(
            self, "VigiaAgentsStateTable",
            table_name="vigia-agents-state",
            partition_key=dynamodb.Attribute(
                name="batman_token",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="agent_timestamp",
                type=dynamodb.AttributeType.NUMBER
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            stream=dynamodb.StreamViewType.NEW_AND_OLD_IMAGES,
            removal_policy=RemovalPolicy.DESTROY
        )

        # Medical audit trail table (HIPAA compliance)
        self.medical_audit_table = dynamodb.Table(
            self, "VigiaAuditTable", 
            table_name="vigia-medical-audit",
            partition_key=dynamodb.Attribute(
                name="audit_id",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="audit_timestamp",
                type=dynamodb.AttributeType.NUMBER
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY
        )

        # LPP detection results table
        self.lpp_results_table = dynamodb.Table(
            self, "VigiaLPPResultsTable",
            table_name="vigia-lpp-results", 
            partition_key=dynamodb.Attribute(
                name="case_id",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="analysis_timestamp",
                type=dynamodb.AttributeType.NUMBER
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY
        )

        # =============================================================================
        # LAMBDA FUNCTIONS - 9-AGENT MEDICAL SYSTEM
        # =============================================================================

        # Base Lambda execution role with medical permissions
        lambda_role = iam.Role(
            self, "VigiaLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
            ]
        )

        # Grant DynamoDB permissions
        self.agents_state_table.grant_read_write_data(lambda_role)
        self.medical_audit_table.grant_read_write_data(lambda_role)
        self.lpp_results_table.grant_read_write_data(lambda_role)
        self.medical_storage.grant_read_write(lambda_role)

        # Master Medical Orchestrator - Central coordination
        self.master_orchestrator = _lambda.Function(
            self, "MasterMedicalOrchestrator",
            function_name="vigia-master-orchestrator",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="master_orchestrator.handler",
            code=_lambda.Code.from_asset("../lambda/master_orchestrator"),
            timeout=Duration.minutes(15),
            memory_size=3008,
            role=lambda_role,
            environment={
                "MEDICAL_STORAGE_BUCKET": self.medical_storage.bucket_name,
                "AGENTS_STATE_TABLE": self.agents_state_table.table_name,
                "MEDICAL_AUDIT_TABLE": self.medical_audit_table.table_name,
                "LPP_RESULTS_TABLE": self.lpp_results_table.table_name
            }
        )

        # Image Analysis Agent - MONAI + YOLOv5 medical imaging
        self.image_analysis_agent = _lambda.Function(
            self, "ImageAnalysisAgent", 
            function_name="vigia-image-analysis",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="image_analysis.handler",
            code=_lambda.Code.from_asset("../lambda/image_analysis"),
            timeout=Duration.minutes(10),
            memory_size=3008,
            role=lambda_role,
            environment={
                "MEDICAL_STORAGE_BUCKET": self.medical_storage.bucket_name,
                "AGENTS_STATE_TABLE": self.agents_state_table.table_name,
                "LPP_RESULTS_TABLE": self.lpp_results_table.table_name
            }
        )

        # Voice Analysis Agent - Hume AI integration
        self.voice_analysis_agent = _lambda.Function(
            self, "VoiceAnalysisAgent",
            function_name="vigia-voice-analysis", 
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="voice_analysis.handler",
            code=_lambda.Code.from_asset("../lambda/voice_analysis"),
            timeout=Duration.minutes(8),
            memory_size=2048,
            role=lambda_role,
            environment={
                "AGENTS_STATE_TABLE": self.agents_state_table.table_name,
                "MEDICAL_AUDIT_TABLE": self.medical_audit_table.table_name
            }
        )

        # Clinical Assessment Agent - Evidence-based medical evaluation
        self.clinical_assessment_agent = _lambda.Function(
            self, "ClinicalAssessmentAgent",
            function_name="vigia-clinical-assessment",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="clinical_assessment.handler",
            code=_lambda.Code.from_asset("../lambda/clinical_assessment"),
            timeout=Duration.minutes(5),
            memory_size=1024,
            role=lambda_role,
            environment={
                "AGENTS_STATE_TABLE": self.agents_state_table.table_name,
                "LPP_RESULTS_TABLE": self.lpp_results_table.table_name
            }
        )

        # Risk Assessment Agent - Medical risk stratification
        self.risk_assessment_agent = _lambda.Function(
            self, "RiskAssessmentAgent",
            function_name="vigia-risk-assessment",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="risk_assessment.handler", 
            code=_lambda.Code.from_asset("../lambda/risk_assessment"),
            timeout=Duration.minutes(5),
            memory_size=1024,
            role=lambda_role,
            environment={
                "AGENTS_STATE_TABLE": self.agents_state_table.table_name,
                "LPP_RESULTS_TABLE": self.lpp_results_table.table_name
            }
        )

        # Diagnostic Agent - Multi-agent diagnostic fusion
        self.diagnostic_agent = _lambda.Function(
            self, "DiagnosticAgent",
            function_name="vigia-diagnostic",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="diagnostic.handler",
            code=_lambda.Code.from_asset("../lambda/diagnostic"),
            timeout=Duration.minutes(8),
            memory_size=2048,
            role=lambda_role,
            environment={
                "AGENTS_STATE_TABLE": self.agents_state_table.table_name,
                "LPP_RESULTS_TABLE": self.lpp_results_table.table_name,
                "MEDICAL_AUDIT_TABLE": self.medical_audit_table.table_name
            }
        )

        # Protocol Agent - NPUAP/EPUAP/PPPIA 2019 guidelines
        self.protocol_agent = _lambda.Function(
            self, "ProtocolAgent",
            function_name="vigia-protocol",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="protocol.handler",
            code=_lambda.Code.from_asset("../lambda/protocol"),
            timeout=Duration.minutes(5),
            memory_size=1024,
            role=lambda_role,
            environment={
                "AGENTS_STATE_TABLE": self.agents_state_table.table_name,
                "LPP_RESULTS_TABLE": self.lpp_results_table.table_name
            }
        )

        # Communication Agent - WhatsApp/Slack coordination
        self.communication_agent = _lambda.Function(
            self, "CommunicationAgent",
            function_name="vigia-communication",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="communication.handler",
            code=_lambda.Code.from_asset("../lambda/communication"),
            timeout=Duration.minutes(5),
            memory_size=1024,
            role=lambda_role,
            environment={
                "AGENTS_STATE_TABLE": self.agents_state_table.table_name,
                "MEDICAL_AUDIT_TABLE": self.medical_audit_table.table_name
            }
        )

        # MONAI Review Agent - Medical imaging quality assessment
        self.monai_review_agent = _lambda.Function(
            self, "MonaiReviewAgent",
            function_name="vigia-monai-review",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="monai_review.handler",
            code=_lambda.Code.from_asset("../lambda/monai_review"),
            timeout=Duration.minutes(10),
            memory_size=3008,
            role=lambda_role,
            environment={
                "MEDICAL_STORAGE_BUCKET": self.medical_storage.bucket_name,
                "AGENTS_STATE_TABLE": self.agents_state_table.table_name,
                "LPP_RESULTS_TABLE": self.lpp_results_table.table_name
            }
        )

        # =============================================================================
        # STEP FUNCTIONS - 9-AGENT ORCHESTRATION WORKFLOW
        # =============================================================================

        # Master Orchestrator Task
        master_orchestration_task = sfn_tasks.LambdaInvoke(
            self, "MasterOrchestrationTask",
            lambda_function=self.master_orchestrator,
            payload=sfn.TaskInput.from_json_path_at("$.medical_case"),
            result_path="$.orchestration_result"
        )

        # Image Analysis Task
        image_analysis_task = sfn_tasks.LambdaInvoke(
            self, "ImageAnalysisTask",
            lambda_function=self.image_analysis_agent,
            payload=sfn.TaskInput.from_json_path_at("$.image_data"),
            result_path="$.image_analysis_result"
        )

        # Voice Analysis Task
        voice_analysis_task = sfn_tasks.LambdaInvoke(
            self, "VoiceAnalysisTask",
            lambda_function=self.voice_analysis_agent,
            payload=sfn.TaskInput.from_json_path_at("$.voice_data"),
            result_path="$.voice_analysis_result"
        )

        # Clinical Assessment Task
        clinical_assessment_task = sfn_tasks.LambdaInvoke(
            self, "ClinicalAssessmentTask",
            lambda_function=self.clinical_assessment_agent,
            payload=sfn.TaskInput.from_json_path_at("$.clinical_data"),
            result_path="$.clinical_result"
        )

        # Risk Assessment Task
        risk_assessment_task = sfn_tasks.LambdaInvoke(
            self, "RiskAssessmentTask",
            lambda_function=self.risk_assessment_agent,
            payload=sfn.TaskInput.from_json_path_at("$.risk_data"),
            result_path="$.risk_result"
        )

        # MONAI Review Task
        monai_review_task = sfn_tasks.LambdaInvoke(
            self, "MonaiReviewTask",
            lambda_function=self.monai_review_agent,
            payload=sfn.TaskInput.from_json_path_at("$.monai_data"),
            result_path="$.monai_result"
        )

        # Diagnostic Fusion Task
        diagnostic_task = sfn_tasks.LambdaInvoke(
            self, "DiagnosticTask",
            lambda_function=self.diagnostic_agent,
            payload=sfn.TaskInput.from_json_path_at("$.diagnostic_data"),
            result_path="$.diagnostic_result"
        )

        # Protocol Validation Task
        protocol_task = sfn_tasks.LambdaInvoke(
            self, "ProtocolTask",
            lambda_function=self.protocol_agent,
            payload=sfn.TaskInput.from_json_path_at("$.protocol_data"),
            result_path="$.protocol_result"
        )

        # Communication Task
        communication_task = sfn_tasks.LambdaInvoke(
            self, "CommunicationTask",
            lambda_function=self.communication_agent,
            payload=sfn.TaskInput.from_json_path_at("$.communication_data")
        )

        # Parallel processing for multimodal analysis
        multimodal_processing = sfn.Parallel(
            self, "MultimodalProcessing",
            comment="Execute image and voice analysis in parallel"
        )

        multimodal_processing.branch(
            sfn.Chain.start(image_analysis_task)
        )
        multimodal_processing.branch(
            sfn.Chain.start(voice_analysis_task)
        )
        multimodal_processing.branch(
            sfn.Chain.start(monai_review_task)
        )

        # Clinical analysis parallel processing
        clinical_processing = sfn.Parallel(
            self, "ClinicalProcessing",
            comment="Execute clinical and risk assessment in parallel"
        )

        clinical_processing.branch(
            sfn.Chain.start(clinical_assessment_task)
        )
        clinical_processing.branch(
            sfn.Chain.start(risk_assessment_task)
        )

        # Sequential workflow: Master → Multimodal → Clinical → Diagnostic → Protocol → Communication
        definition = (
            master_orchestration_task
            .next(multimodal_processing)
            .next(clinical_processing)
            .next(diagnostic_task)
            .next(protocol_task)
            .next(communication_task)
        )

        # Step Functions State Machine
        self.vigia_workflow = sfn.StateMachine(
            self, "VigiaWorkflow",
            state_machine_name="vigia-9-agent-workflow",
            definition=definition,
            timeout=Duration.minutes(45)
        )

        # =============================================================================
        # FASTAPI WEB INTERFACE - LAMBDA CONTAINER
        # =============================================================================
        
        # FastAPI web interface as Lambda container (Optimized deployment)
        self.fastapi_lambda = _lambda.Function(
            self, "VigiaFastAPILambda",
            function_name="vigia-fastapi-web-interface",
            code=_lambda.Code.from_ecr_image(
                repository=ecr.Repository.from_repository_name(
                    self, "VigiaFastAPIRepo",
                    repository_name="vigia-fastapi"
                ),
                tag_or_digest="lambda-single"  # Using single-platform Lambda container
            ),
            handler=_lambda.Handler.FROM_IMAGE,
            runtime=_lambda.Runtime.FROM_IMAGE,
            memory_size=2048,  # Optimized memory for web interface
            timeout=Duration.seconds(30),  # API Gateway max timeout
            environment={
                "AWS_DEPLOYMENT": "true",
                "LAMBDA_DEPLOYMENT": "true",
                "VIGIA_ENV": "production",
                "MEDICAL_MODE": "production",
                "PHI_TOKENIZATION_ENABLED": "true",
                "AGENTS_STATE_TABLE": self.agents_state_table.table_name,
                "MEDICAL_AUDIT_TABLE": self.medical_audit_table.table_name,
                "MEDICAL_STORAGE_BUCKET": self.medical_storage.bucket_name,
                "WORKFLOW_ARN": self.vigia_workflow.state_machine_arn,
                # AWS MCP patterns for better monitoring
                "AWS_XRAY_TRACING_NAME": "vigia-fastapi-web",
                "AWS_LAMBDA_POWERTOOLS_SERVICE_NAME": "vigia-medical-ai",
                "AWS_LAMBDA_POWERTOOLS_METRICS_NAMESPACE": "VIGIA/Medical",
                "LOG_LEVEL": "INFO"
            },
            vpc=None,  # No VPC for web interface
            description="VIGIA Medical AI FastAPI Web Interface - Optimized Container",
            # AWS MCP patterns for reliability
            tracing=_lambda.Tracing.ACTIVE,  # Enable X-Ray tracing
            retry_attempts=0,  # Disable automatic retries for web interface
            # Dead letter queue for failed invocations
            dead_letter_queue_enabled=True,
            # Architecture optimization
            architecture=_lambda.Architecture.X86_64
        )
        
        # Grant permissions to access medical resources
        self.agents_state_table.grant_read_write_data(self.fastapi_lambda)
        self.medical_audit_table.grant_read_write_data(self.fastapi_lambda)
        self.medical_storage.grant_read_write(self.fastapi_lambda)
        self.vigia_workflow.grant_start_execution(self.fastapi_lambda)
        
        # Create Function URL for direct access
        self.fastapi_function_url = self.fastapi_lambda.add_function_url(
            auth_type=_lambda.FunctionUrlAuthType.NONE,
            cors=_lambda.FunctionUrlCorsOptions(
                allowed_origins=["*"],
                allowed_methods=[_lambda.HttpMethod.ALL],
                allowed_headers=["*"]
            )
        )

        # =============================================================================
        # CUSTOM DOMAIN PLACEHOLDER - MANUAL SETUP REQUIRED
        # =============================================================================
        
        # Note: Custom domain setup requires:
        # 1. Register autonomos.dev domain in Route 53
        # 2. Create hosted zone for autonomos.dev
        # 3. Deploy CloudFront distribution manually
        # For now, using direct Lambda Function URL
        
        custom_domain_configured = False
        distribution = None
        certificate = None

        # =============================================================================
        # MONITORING & ALERTING - AWS MCP PATTERNS
        # =============================================================================
        
        # SNS Topic for medical system alerts
        self.medical_alerts_topic = sns.Topic(
            self, "VigiamedicalAlerts",
            topic_name="vigia-medical-alerts",
            display_name="VIGIA Medical AI System Alerts"
        )
        
        # CloudWatch Log Groups with structured logging
        self.fastapi_log_group = logs.LogGroup(
            self, "VigiaFastAPILogGroup",
            log_group_name=f"/aws/lambda/{self.fastapi_lambda.function_name}",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY
        )
        
        # CloudWatch Alarms for FastAPI Lambda
        self.fastapi_error_alarm = cloudwatch.Alarm(
            self, "VigiaFastAPIErrorAlarm",
            alarm_name="vigia-fastapi-error-rate",
            alarm_description="VIGIA FastAPI Lambda error rate is high",
            metric=self.fastapi_lambda.metric_errors(
                period=Duration.minutes(5),
                statistic="Sum"
            ),
            threshold=5,
            evaluation_periods=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD
        )
        
        # Add SNS notification to alarm
        self.fastapi_error_alarm.add_alarm_action(
            cw_actions.SnsAction(self.medical_alerts_topic)
        )
        
        # Duration alarm for performance monitoring
        self.fastapi_duration_alarm = cloudwatch.Alarm(
            self, "VigiaFastAPIDurationAlarm",
            alarm_name="vigia-fastapi-high-duration",
            alarm_description="VIGIA FastAPI Lambda duration is high",
            metric=self.fastapi_lambda.metric_duration(
                period=Duration.minutes(5),
                statistic="Average"
            ),
            threshold=25000,  # 25 seconds (close to 30s timeout)
            evaluation_periods=3,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD
        )
        
        self.fastapi_duration_alarm.add_alarm_action(
            cw_actions.SnsAction(self.medical_alerts_topic)
        )
        
        # Medical workflow error monitoring
        self.workflow_error_alarm = cloudwatch.Alarm(
            self, "VigiaWorkflowErrorAlarm",
            alarm_name="vigia-workflow-execution-failures",
            alarm_description="VIGIA Medical workflow has execution failures",
            metric=cloudwatch.Metric(
                namespace="AWS/States",
                metric_name="ExecutionsFailed",
                dimensions_map={
                    "StateMachineArn": self.vigia_workflow.state_machine_arn
                },
                period=Duration.minutes(5),
                statistic="Sum"
            ),
            threshold=1,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD
        )
        
        self.workflow_error_alarm.add_alarm_action(
            cw_actions.SnsAction(self.medical_alerts_topic)
        )
        
        # Medical system dashboard
        self.medical_dashboard = cloudwatch.Dashboard(
            self, "VigiamedicalDashboard",
            dashboard_name="vigia-medical-ai-monitoring",
            widgets=[
                [
                    cloudwatch.GraphWidget(
                        title="FastAPI Lambda Performance",
                        left=[self.fastapi_lambda.metric_invocations()],
                        right=[self.fastapi_lambda.metric_errors()],
                        width=12,
                        height=6
                    )
                ],
                [
                    cloudwatch.GraphWidget(
                        title="FastAPI Lambda Duration",
                        left=[self.fastapi_lambda.metric_duration()],
                        width=12,
                        height=6
                    )
                ],
                [
                    cloudwatch.GraphWidget(
                        title="Medical Workflow Executions",
                        left=[
                            cloudwatch.Metric(
                                namespace="AWS/States",
                                metric_name="ExecutionsStarted",
                                dimensions_map={
                                    "StateMachineArn": self.vigia_workflow.state_machine_arn
                                }
                            ),
                            cloudwatch.Metric(
                                namespace="AWS/States", 
                                metric_name="ExecutionsSucceeded",
                                dimensions_map={
                                    "StateMachineArn": self.vigia_workflow.state_machine_arn
                                }
                            )
                        ],
                        right=[
                            cloudwatch.Metric(
                                namespace="AWS/States",
                                metric_name="ExecutionsFailed",
                                dimensions_map={
                                    "StateMachineArn": self.vigia_workflow.state_machine_arn
                                }
                            )
                        ],
                        width=12,
                        height=6
                    )
                ]
            ]
        )

        # =============================================================================
        # API GATEWAY - MEDICAL INTERFACES
        # =============================================================================

        # REST API for medical system integration
        self.api = apigateway.RestApi(
            self, "VigiaAPI",
            rest_api_name="vigia-medical-api",
            description="VIGIA Medical AI - 9-Agent Pressure Injury Detection API",
            default_cors_preflight_options=apigateway.CorsOptions(
                allow_origins=apigateway.Cors.ALL_ORIGINS,
                allow_methods=apigateway.Cors.ALL_METHODS,
                allow_headers=["Content-Type", "Authorization", "X-Medical-Token"]
            )
        )

        # Medical workflow trigger endpoint
        medical_workflow_integration = apigateway.AwsIntegration(
            service="states",
            action="StartExecution",
            integration_http_method="POST",
            options=apigateway.IntegrationOptions(
                credentials_role=iam.Role(
                    self, "APIGatewayMedicalRole",
                    assumed_by=iam.ServicePrincipal("apigateway.amazonaws.com"),
                    managed_policies=[
                        iam.ManagedPolicy.from_aws_managed_policy_name("AWSStepFunctionsFullAccess")
                    ]
                ),
                request_templates={
                    "application/json": f'''{{
                        "stateMachineArn": "{self.vigia_workflow.state_machine_arn}",
                        "input": "$util.escapeJavaScript($input.body)"
                    }}'''
                }
            )
        )

        self.api.root.add_resource("medical").add_resource("analyze").add_method("POST", medical_workflow_integration)

        # Health check endpoint
        health_integration = apigateway.MockIntegration(
            integration_responses=[
                apigateway.IntegrationResponse(
                    status_code="200",
                    response_templates={
                        "application/json": '{"status": "healthy", "service": "vigia-medical-ai"}'
                    }
                )
            ],
            request_templates={
                "application/json": '{"statusCode": 200}'
            }
        )

        self.api.root.add_resource("health").add_method(
            "GET", 
            health_integration,
            method_responses=[
                apigateway.MethodResponse(status_code="200")
            ]
        )

        # =============================================================================
        # EVENTBRIDGE - SCHEDULED MEDICAL PROCESSING
        # =============================================================================

        # Daily medical system health check
        daily_health_rule = events.Rule(
            self, "DailyHealthRule",
            schedule=events.Schedule.cron(hour="7", minute="0"),  # 7 AM daily
            description="Trigger daily VIGIA medical system health check"
        )

        daily_health_rule.add_target(
            targets.SfnStateMachine(
                self.vigia_workflow,
                input=events.RuleTargetInput.from_object({
                    "trigger": "scheduled",
                    "type": "health_check",
                    "medical_case": {
                        "case_type": "system_validation",
                        "batman_token": "SYSTEM-HEALTH-CHECK"
                    }
                })
            )
        )

        # =============================================================================
        # OUTPUTS
        # =============================================================================

        CfnOutput(
            self, "MedicalStorageBucket",
            value=self.medical_storage.bucket_name,
            description="S3 bucket for VIGIA medical imaging storage"
        )

        CfnOutput(
            self, "MedicalAPIEndpoint", 
            value=self.api.url,
            description="VIGIA Medical API Gateway endpoint"
        )
        
        CfnOutput(
            self, "FastAPIWebInterface",
            value=self.fastapi_function_url.url,
            description="VIGIA Medical AI FastAPI Web Interface URL"
        )

        CfnOutput(
            self, "MedicalWorkflowArn",
            value=self.vigia_workflow.state_machine_arn,
            description="VIGIA 9-Agent Step Functions workflow ARN"
        )

        CfnOutput(
            self, "AgentsStateTableOutput",
            value=self.agents_state_table.table_name,
            description="DynamoDB table for VIGIA agent state management"
        )

        CfnOutput(
            self, "MedicalAuditTableOutput", 
            value=self.medical_audit_table.table_name,
            description="DynamoDB table for HIPAA medical audit trail"
        )

        CfnOutput(
            self, "LPPResultsTableOutput",
            value=self.lpp_results_table.table_name,
            description="DynamoDB table for LPP detection results"
        )

        CfnOutput(
            self, "MasterOrchestratorOutput",
            value=self.master_orchestrator.function_name,
            description="Lambda function for VIGIA master medical orchestrator"
        )
        
        CfnOutput(
            self, "MedicalAlertsTopicOutput",
            value=self.medical_alerts_topic.topic_arn,
            description="SNS topic for VIGIA medical system alerts"
        )
        
        CfnOutput(
            self, "MedicalDashboardOutput",
            value=f"https://console.aws.amazon.com/cloudwatch/home?region={self.region}#dashboards:name={self.medical_dashboard.dashboard_name}",
            description="CloudWatch dashboard for VIGIA medical system monitoring"
        )
        
        CfnOutput(
            self, "OptimizedContainerImageOutput",
            value="586794472237.dkr.ecr.us-east-1.amazonaws.com/vigia-fastapi:autonomos-dev",
            description="ECR URI for autonomos.dev VIGIA FastAPI container"
        )
        
        # Custom domain outputs (conditional)
        if 'custom_domain_configured' in locals() and custom_domain_configured:
            CfnOutput(
                self, "CustomDomainURL",
                value="https://www.autonomos.dev",
                description="Custom domain URL for VIGIA Medical AI"
            )
            
            CfnOutput(
                self, "CloudFrontDistributionID",
                value=distribution.distribution_id,
                description="CloudFront distribution ID for autonomos.dev"
            )
            
            CfnOutput(
                self, "SSLCertificateArn",
                value=certificate.certificate_arn,
                description="SSL certificate ARN for autonomos.dev"
            )