# 📊 VIGIA Medical AI v1.0 - Architecture Diagrams (CORRECTED)

## 🎯 Objetivo

Collection of 6 corrected diagrams showing **real VIGIA Medical AI production system**. Each diagram reflects the **actual deployed architecture** with AWS CDK: 9 Lambda agents + Batman PHI tokenization + AgentOps monitoring + professional domains (autonomos.dev). All diagrams fixed to eliminate architectural inconsistencies.

---

## 📋 Índice de Diagramas

### 1. 🏗️ [System Overview](./01_system_overview.eraser)
**⏱️ Reading time: 5 minutes**

**When to use:**
- New developers joining project
- Executive presentations  
- Medical-technical onboarding
- System demos

**What it shows:**
- **Real deployed system**: 9 Lambda agents (realistic 1-3GB memory) with Step Functions
- **Professional domains**: autonomos.dev + vigia.autonomos.dev with SSL certificates
- **Batman PHI tokenization**: Bruce Wayne → Batman tokens, zero PHI in processing
- **AgentOps monitoring**: HIPAA-compliant telemetry integrated throughout
- **Cost efficiency**: $200-500/month serverless vs $1,275+ containers

---

### 2. ⚙️ [Technical Architecture](./02_technical_architecture.eraser)
**⏱️ Reading time: 8 minutes**

**When to use:**
- Technical architecture decisions
- Code reviews
- Scalability planning
- Technology stack evaluation

**What it shows:**
- **Simplified AWS stack**: No artificial layers, real component groups
- **9 Lambda agents**: Real memory allocations (1-3GB) without duplications
- **Unified monitoring**: AgentOps primary + CloudWatch backup
- **External AI clear**: Each service used by specific agents
- **Direct workflow**: Step Functions coordination without complex connections

---

### 3. 🔄 [Critical User Flow](./03_critical_user_flow.eraser)
**⏱️ Reading time: 7 minutes**

**When to use:**
- Critical flow testing
- Medical experience debugging
- Performance optimization
- Medical process validation

**What it shows:**
- **Pure user experience**: Patient → Analysis → Medical response (no technical details)
- **Realistic communication**: "Dr. Smith en camino" vs "batman_TC001"
- **3-phase flow**: Input → Process → Output without parallel/sequential confusion
- **Human actors**: Patient, WhatsApp, VIGIA System, Dr. Smith, Medical Team
- **Timeline**: 3 minutes total from image to medical response

---

### 4. 💾 [Data Model](./04_data_model.eraser)
**⏱️ Reading time: 6 minutes**

**When to use:**
- Database schema updates
- HIPAA compliance audits
- Data migration planning
- PHI security reviews

**What it shows:**
- **Batman tokenization focus**: Bruce Wayne → Batman tokens with 15-minute TTL
- **3 DynamoDB tables**: Real CDK deployment with batman_token primary keys
- **KMS encryption**: All medical data encrypted with AWS KMS keys
- **HIPAA compliance**: Complete audit trail with tokenization method tracking
- **Multi-agent results**: 9 agents write analysis results to structured tables

---

### 5. 🔌 [External Integrations](./05_external_integrations.eraser)
**⏱️ Reading time: 5 minutes**

**When to use:**
- Dependency risk assessment
- SLA planning and monitoring
- Backup strategy design
- Cost optimization analysis

**What it shows:**
- **Real external services**: Hume AI (92% pain detection), AWS Bedrock, AgentOps
- **Cost analysis**: $225-581/month vs $1,275-2,020 containers (82% savings)
- **Risk mitigation**: Local fallbacks for all external dependencies
- **Communication integration**: WhatsApp patient + Slack medical team
- **Professional domains**: autonomos.dev + vigia.autonomos.dev with SSL

---

### 6. 🚀 [Deployment Infrastructure](./06_deployment_infrastructure.eraser)
**⏱️ Reading time: 8 minutes**

**When to use:**
- Production deployment planning
- DevOps configuration
- Infrastructure scaling
- Security hardening

**What it shows:**
- **Real CDK deployment**: vigia_stack.py with 863 lines, 9 Lambda functions
- **Professional domains**: autonomos.dev + vigia.autonomos.dev with SSL certificates
- **Container deployment**: FastAPI optimized containers on Lambda
- **AgentOps monitoring**: HIPAA-compliant telemetry for all 9 agents
- **Cost efficiency**: $200-500/month serverless infrastructure

---

## ❌ Diagram Removed
**07_entity_states.eraser** was removed during architectural correction as it contained redundant state information already covered in the data model and technical architecture diagrams.

---

## 🛠️ How to Use These Diagrams

### 📖 For Reading
1. **New to project**: Start with `01_system_overview`
2. **Technical developers**: Continue with `02_technical_architecture` 
3. **Testing/QA**: Focus on `03_critical_user_flow`
4. **Database/Data**: Review `04_data_model`
5. **DevOps/Infrastructure**: Study `06_deployment_infrastructure`

### ☁️ For AWS Deployment Context
1. **Production Architecture**: All diagrams reflect real AWS serverless deployment
2. **Cost Analysis**: $200-500/month serverless vs $1,275-2,020/month containers
3. **Domain Configuration**: autonomos.dev live with S3 + CloudFront + Route53
4. **CDK Deployment**: Infrastructure-as-code with `cdk deploy VigiaStack`

### ✏️ For Editing in Eraser.io
1. **Open any `.eraser` file**
2. **Copy the code content**
3. **Go to [eraser.io](https://eraser.io)**
4. **Create new diagram** → Diagram-as-code
5. **Paste code** and edit visually
6. **Export** or save changes

### 🔄 For Updates
When making significant code changes:

1. **Identify affected diagrams**:
   - New Lambda agents → `01_system_overview` + `02_technical_architecture`
   - New AWS services → `05_external_integrations` + `06_deployment_infrastructure`
   - DynamoDB schema changes → `04_data_model`
   - Step Functions workflow changes → `03_critical_user_flow`
   - CDK infrastructure changes → `06_deployment_infrastructure`

2. **Update Eraser.io code**:
   - Maintain exact Eraser.io syntax
   - Use only available icons
   - Preserve consistent colors and styling

3. **Validate functionality**:
   - Test code in eraser.io
   - Verify compilation without errors
   - Maintain visual clarity

---

## 🎨 Eraser.io Style Guide

### **VIGIA + AWS Standard Icons**
```
- Medical: medical, stethoscope, hospital
- Users: user, users, admin
- AI/Brain: brain, eye, hub
- Security: shield, lock, key
- Communication: mobile, slack, mail
- Data: database, cache, storage
- AWS Serverless: aws-lambda, aws-stepfunctions, aws-dynamodb, aws-apigateway
- AWS Storage: aws-s3, aws-cloudwatch, aws-cloudfront, aws-route53
- AWS AI: aws-bedrock, aws-sagemaker
- AWS Core: aws, aws-cloudformation, aws-cdk, aws-iam
- Cloud: cloud, aws, gcp
- Monitoring: monitor, chart, graph
```

### **Consistent Colors**
```
- Medical Critical: red
- AI/Processing: purple
- Security/PHI: red
- Communication: green
- Data Storage: blue  
- Monitoring: orange
- External Services: gray
```

### **Syntax Patterns**
```
// Service groups
Group_Name [icon: icon, color: color] {
  Service1 [icon: service_icon]
  Service2 [icon: service_icon]
}

// Connections with context
Service1 > Service2: "Action description"
Service1 <> Service2: "Bidirectional"

// Critical properties
Service [icon: icon, color: color] {
  risk_level: "HIGH"
  uptime: "99.9%"
  backup: "Fallback system"
}
```

---

## 🚨 Maintenance Guidelines

### **When to Update (Production v1.0)**
- ✅ **New Lambda medical agents**: Update `01_system_overview` + `02_technical_architecture`
- ✅ **DynamoDB schema changes**: Update `04_data_model`
- ✅ **AgentOps integration changes**: Update ALL diagrams (integrated throughout)
- ✅ **Domain/SSL updates**: Update `06_deployment_infrastructure` + `05_external_integrations`
- ✅ **Step Functions workflow modifications**: Update `03_critical_user_flow`
- ✅ **Multimodal analysis features**: Update `03_critical_user_flow` + `01_system_overview`
- ✅ **Cost optimization updates**: Update `05_external_integrations` (real production costs)

### **Validation Checklist**
- [ ] Code compiles in eraser.io without errors
- [ ] Icons are valid and available
- [ ] Colors follow style guide
- [ ] Connections have descriptive labels
- [ ] Notes include critical information
- [ ] Timing estimates are realistic

---

## 📞 Support

**Questions about diagrams?**
- 📖 **Docs**: Read main `README.md`
- 🔧 **Technical**: Check `CLAUDE.md` for development context
- 🏥 **Medical**: Consult `docs/ARCHITECTURE.md` for medical details

**To contribute:**
1. Update diagrams using exact Eraser.io syntax
2. Maintain consistency with style guide
3. Validate functionality in eraser.io
4. Update this README if adding new diagrams

---

**🩺 VIGIA Medical AI v1.0 - Corrected diagrams reflect real deployed system.**  
**☁️ Live at autonomos.dev + vigia.autonomos.dev with 9 Lambda agents + AgentOps monitoring.**  
**📊 Updated 2025-07-04 - All 6 diagrams corrected to eliminate architectural inconsistencies.**  
**✅ Fixed: Realistic Lambda memory, Batman PHI tokenization, unified monitoring, simplified flows.**