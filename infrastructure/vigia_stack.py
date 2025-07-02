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