# 📊 VIGIA Medical AI - Architecture Diagrams

## 🎯 Objetivo

Esta colección de 7 diagramas esenciales te permite entender **VIGIA Medical AI** en 30 minutos máximo. Cada diagrama está optimized usando sintaxis nativa de Eraser.io para máxima claridad y facilidad de edición.

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
- Arquitectura médica completa de alto nivel
- PHI tokenización (Bruce Wayne → Batman)
- 9 agentes especializados coordinados
- Comunicación bidireccional WhatsApp ↔ Slack

---

### 2. ⚙️ [Technical Architecture](./02_technical_architecture.eraser)
**⏱️ Tiempo de comprensión: 8 minutos**

**Cuándo usar:**
- Decisiones de arquitectura técnica
- Code reviews estructurales
- Planning de escalabilidad
- Evaluación de stack tecnológico

**Qué muestra:**
- 3-Layer Security Design detallado
- Google Cloud ADK integration
- Medical AI Stack (MONAI + PyTorch + YOLOv5)
- HIPAA compliance architecture

---

### 3. 🔄 [Critical User Flow](./03_critical_user_flow.eraser)
**⏱️ Tiempo de comprensión: 7 minutos**

**Cuándo usar:**
- Testing de flujos críticos
- Debugging de experiencia médica
- Optimización de performance
- Validación de procesos médicos

**Qué muestra:**
- Happy path completo: paciente → AI analysis → equipo médico
- Análisis paralelo de 9 agentes médicos
- Grade 4 LPP emergency escalation
- Sub-3 minutos response time

---

### 4. 💾 [Data Model](./04_data_model.eraser)
**⏱️ Tiempo de comprensión: 6 minutos**

**Cuándo usar:**
- Database schema updates
- HIPAA compliance audits
- Data migration planning
- PHI security reviews

**Qué muestra:**
- PHI tokenización database design
- Medical session management
- Agent analysis results storage
- Complete audit trail structure

---

### 5. 🔌 [External Integrations](./05_external_integrations.eraser)
**⏱️ Tiempo de comprensión: 5 minutos**

**Cuándo usar:**
- Risk assessment de dependencias
- SLA planning y monitoring
- Backup strategy design
- Cost optimization analysis

**Qué muestra:**
- Google Cloud ADK (HIGH RISK SPOF)
- Communication APIs (Twilio + Slack)
- Local AI stack reliability
- Fallback mechanisms

---

### 6. 🚀 [Deployment Infrastructure](./06_deployment_infrastructure.eraser)
**⏱️ Tiempo de comprensión: 8 minutos**

**Cuándo usar:**
- Production deployment planning
- DevOps configuration
- Infrastructure scaling
- Security hardening

**Qué muestra:**
- Local-first development (`./install.sh`)
- Production cloud options (Render + alternatives)
- Medical-grade containerization
- HIPAA infrastructure compliance

---

### 7. 📊 [Entity States](./07_entity_states.eraser)
**⏱️ Tiempo de comprensión: 6 minutos**

**Cuándo usar:**
- State machine debugging
- Medical workflow optimization
- Timeout management tuning
- Emergency escalation testing

**Qué muestra:**
- Medical session lifecycle (15-min timeout)
- Emergency state transitions (Grade 4 → 30-min extension)
- 9-agent coordination states
- HIPAA audit trail completion

---

## 🛠️ Cómo Usar Estos Diagramas

### 📖 Para Lectura
1. **Nuevos en el proyecto**: Empieza con `01_system_overview`
2. **Developers técnicos**: Continúa con `02_technical_architecture` 
3. **Testing/QA**: Focus en `03_critical_user_flow`
4. **Database/Data**: Revisa `04_data_model`
5. **DevOps/Infrastructure**: Estudia `06_deployment_infrastructure`

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
   - New agents → `01_system_overview` + `02_technical_architecture`
   - New API integrations → `05_external_integrations`
   - Database changes → `04_data_model`
   - Workflow changes → `03_critical_user_flow` + `07_entity_states`

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

### **Iconos Estándar VIGIA**
```
- Medical: medical, stethoscope, hospital
- Users: user, users, admin
- AI/Brain: brain, eye, hub
- Security: shield, lock, key
- Communication: mobile, slack, mail
- Data: database, cache, storage
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

### **Cuando Actualizar**
- ✅ **New medical agents added**: Update `01_system_overview`
- ✅ **Database schema changes**: Update `04_data_model`
- ✅ **New external APIs**: Update `05_external_integrations`
- ✅ **Workflow modifications**: Update `03_critical_user_flow`
- ✅ **Infrastructure changes**: Update `06_deployment_infrastructure`

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

**🩺 Built for medical understanding. Optimized for Eraser.io. Ready for production documentation.**