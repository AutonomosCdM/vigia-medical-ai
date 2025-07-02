# 📊 VIGIA Medical AI - AWS Serverless Architecture Diagrams

## 🎯 Objetivo

Esta colección de 7 diagramas esenciales te permite entender **VIGIA Medical AI AWS Serverless** en 30 minutos máximo. Cada diagrama refleja la arquitectura serverless productiva con Lambda + Step Functions + DynamoDB, optimizado usando sintaxis nativa de Eraser.io para máxima claridad y facilidad de edición.

---

## 📋 Índice de Diagramas

### 1. 🏗️ [System Overview](./01_system_overview.eraser)
**⏱️ Tiempo de comprensión: 5 minutos**

**Cuándo usar:**
- Nuevos desarrolladores en el proyecto
- Presentaciones ejecutivas
- Onboarding médico-técnico
- Demos rápidos del sistema

**Qué muestra:**
- AWS serverless architecture completa (Lambda + Step Functions + DynamoDB)
- PHI tokenización con KMS encryption (Bruce Wayne → Batman)
- 9 Lambda agents especializados con A2A coordination
- Comunicación bidireccional WhatsApp ↔ Slack + autonomos.dev frontend

---

### 2. ⚙️ [Technical Architecture](./02_technical_architecture.eraser)
**⏱️ Tiempo de comprensión: 8 minutos**

**Cuándo usar:**
- Decisiones de arquitectura técnica
- Code reviews estructurales
- Planning de escalabilidad
- Evaluación de stack tecnológico

**Qué muestra:**
- 3-Layer AWS Serverless Security Design (API Gateway → Step Functions → Lambda)
- Step Functions A2A coordination protocol
- AWS AI Stack (Bedrock + SageMaker MONAI + Transcribe Medical)
- HIPAA serverless compliance architecture

---

### 3. 🔄 [Critical User Flow](./03_critical_user_flow.eraser)
**⏱️ Tiempo de comprensión: 7 minutos**

**Cuándo usar:**
- Testing de flujos críticos
- Debugging de experiencia médica
- Optimización de performance
- Validación de procesos médicos

**Qué muestra:**
- AWS serverless happy path: paciente → API Gateway → Step Functions → Lambda agents
- Step Functions parallel execution de 9 Lambda medical agents
- DynamoDB state management con KMS encryption
- Sub-3 minutos serverless response time

---

### 4. 💾 [Data Model](./04_data_model.eraser)
**⏱️ Tiempo de comprensión: 6 minutos**

**Cuándo usar:**
- Database schema updates
- HIPAA compliance audits
- Data migration planning
- PHI security reviews

**Qué muestra:**
- DynamoDB PHI tokenización schema con KMS encryption
- Lambda execution state management con TTL
- S3 + DynamoDB agent analysis results storage
- Complete serverless audit trail structure

---

### 5. 🔌 [External Integrations](./05_external_integrations.eraser)
**⏱️ Tiempo de comprensión: 5 minutos**

**Cuándo usar:**
- Risk assessment de dependencias
- SLA planning y monitoring
- Backup strategy design
- Cost optimization analysis

**Qué muestra:**
- AWS Serverless Services (Lambda, DynamoDB, Step Functions, API Gateway)
- External APIs con Lambda integration (Hume AI + Slack + WhatsApp)
- Cost analysis serverless ($200-500/month vs $1,275-2,020)
- Serverless fallback mechanisms

---

### 6. 🚀 [Deployment Infrastructure](./06_deployment_infrastructure.eraser)
**⏱️ Tiempo de comprensión: 8 minutos**

**Cuándo usar:**
- Production deployment planning
- DevOps configuration
- Infrastructure scaling
- Security hardening

**Qué muestra:**
- AWS CDK infrastructure-as-code deployment (`cdk deploy VigiaStack`)
- Serverless production architecture (Lambda + Step Functions + DynamoDB)
- autonomos.dev domain setup con S3 static website + CloudFront
- HIPAA serverless infrastructure compliance

---

### 7. 📊 [Entity States](./07_entity_states.eraser)
**⏱️ Tiempo de comprensión: 6 minutos**

**Cuándo usar:**
- State machine debugging
- Medical workflow optimization
- Timeout management tuning
- Emergency escalation testing

**Qué muestra:**
- DynamoDB medical session lifecycle con TTL (15-min timeout)
- Step Functions emergency state transitions (Grade 4 → 30-min extension)
- Lambda 9-agent coordination states
- DynamoDB HIPAA audit trail completion

---

## 🛠️ Cómo Usar Estos Diagramas

### 📖 Para Lectura
1. **Nuevos en el proyecto**: Empieza con `01_system_overview`
2. **Developers técnicos**: Continúa con `02_technical_architecture` 
3. **Testing/QA**: Focus en `03_critical_user_flow`
4. **Database/Data**: Revisa `04_data_model`
5. **DevOps/Infrastructure**: Estudia `06_deployment_infrastructure`

### ☁️ Para AWS Deployment Context
1. **Production Architecture**: Todos los diagramas reflejan AWS serverless production
2. **Cost Analysis**: $200-500/month serverless vs $1,275-2,020/month containers
3. **Domain Configuration**: autonomos.dev live con S3 + CloudFront + Route53
4. **CDK Deployment**: Infrastructure-as-code con `cdk deploy VigiaStack`

### ✏️ Para Editar en Eraser.io
1. **Abre cualquier archivo `.eraser`**
2. **Copia el contenido del código**
3. **Ve a [eraser.io](https://eraser.io)**
4. **Crea nuevo diagrama** → Diagram-as-code
5. **Pega el código** y edita visualmente
6. **Exporta** o guarda cambios

### 🔄 Para Actualizar
Cuando hagas cambios significativos al código:

1. **Identifica diagramas afectados**:
   - New Lambda agents → `01_system_overview` + `02_technical_architecture`
   - New AWS services → `05_external_integrations` + `06_deployment_infrastructure`
   - DynamoDB schema changes → `04_data_model`
   - Step Functions workflow changes → `03_critical_user_flow` + `07_entity_states`
   - CDK infrastructure changes → `06_deployment_infrastructure`

2. **Actualiza código Eraser.io**:
   - Mantén sintaxis exacta de Eraser.io
   - Usa iconos disponibles únicamente
   - Preserva colores y styling consistente

3. **Valida funcionamiento**:
   - Prueba código en eraser.io
   - Verifica que compile sin errores
   - Mantén claridad visual

---

## 🎨 Guía de Estilo Eraser.io

### **Iconos Estándar VIGIA + AWS**
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

### **Colores Consistentes**
```
- Medical Critical: red
- AI/Processing: purple
- Security/PHI: red
- Communication: green
- Data Storage: blue  
- Monitoring: orange
- External Services: gray
```

### **Sintaxis Patterns**
```
// Grupos con servicios
Group_Name [icon: icon, color: color] {
  Service1 [icon: service_icon]
  Service2 [icon: service_icon]
}

// Conexiones con contexto
Service1 > Service2: "Action description"
Service1 <> Service2: "Bidirectional"

// Properties críticas
Service [icon: icon, color: color] {
  risk_level: "HIGH"
  uptime: "99.9%"
  backup: "Fallback system"
}
```

---

## 🚨 Maintenance Guidelines

### **Cuando Actualizar (AWS Serverless Context)**
- ✅ **New Lambda medical agents**: Update `01_system_overview` + `02_technical_architecture`
- ✅ **DynamoDB schema changes**: Update `04_data_model` + `07_entity_states`
- ✅ **New AWS services**: Update `05_external_integrations` + `06_deployment_infrastructure`
- ✅ **Step Functions workflow modifications**: Update `03_critical_user_flow`
- ✅ **CDK infrastructure changes**: Update `06_deployment_infrastructure`
- ✅ **Cost optimization updates**: Update `05_external_integrations`
- ✅ **Domain/DNS changes**: Update `06_deployment_infrastructure` + `05_external_integrations`

### **Validation Checklist**
- [ ] Código compila en eraser.io sin errores
- [ ] Iconos son válidos y disponibles
- [ ] Colores siguen guía de estilo
- [ ] Conexiones tienen labels descriptivos
- [ ] Notes incluyen información crítica
- [ ] Timing estimates son realistas

---

## 📞 Support

**¿Questions sobre los diagramas?**
- 📖 **Docs**: Lee el `README.md` principal
- 🔧 **Technical**: Revisa `CLAUDE.md` para development context
- 🏥 **Medical**: Consulta `docs/ARCHITECTURE.md` para detalles médicos

**Para contribuir:**
1. Actualiza diagramas usando sintaxis Eraser.io exacta
2. Mantén consistency con guía de estilo
3. Valida funcionamiento en eraser.io
4. Actualiza este README si añades nuevos diagramas

---

**🩺 Built for medical understanding. AWS serverless production-ready. Optimized for Eraser.io documentation. 
☁️ Deployed at autonomos.dev with CDK infrastructure-as-code.**