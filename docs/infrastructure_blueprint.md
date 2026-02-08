# Data & AI Infrastructure Blueprint — PV Solar Forecast Platform

## 1. Context

A solar-energy company operates a growing portfolio of photovoltaic (PV) parks.
Accurate day-ahead and intra-day power-generation forecasts reduce imbalance costs,
improve grid scheduling, and support energy trading decisions.

This blueprint describes the **target-state production infrastructure** for the PV
Forecast platform, covering data ingestion through model serving and monitoring.

---

## 2. Requirements

### Functional
| # | Requirement | Rationale |
|---|-------------|-----------|
| F1 | Ingest weather forecast data (temperature, radiation, wind, cloud cover, humidity, precipitation) on an hourly schedule | Weather is the primary driver of solar output |
| F2 | Ingest actual generation (SCADA / inverter telemetry) with ≤ 15 min delay | Needed for monitoring, retraining, and lag features |
| F3 | Serve real-time predictions via REST API (< 200 ms p95 latency) | Downstream systems (trading desk, grid operator) call on demand |
| F4 | Retrain the model automatically when data drift is detected **or** on a weekly schedule | Keeps accuracy high as seasons and panel degradation change distributions |
| F5 | Support multi-park forecasting (scale to 100+ parks) | Business will onboard new parks continuously |
| F6 | Track all experiments, model versions, and data versions | Reproducibility and auditability |
| F7 | Alert operations team on drift, latency spikes, or prediction errors | Fast incident response |

### Non-Functional
| # | Requirement | Target |
|---|-------------|--------|
| NF1 | API availability | ≥ 99.5 % uptime |
| NF2 | Prediction latency | < 200 ms p95 |
| NF3 | Model freshness | ≤ 7 days since last successful training |
| NF4 | Data freshness | Weather data ≤ 1 h old; generation data ≤ 15 min old |
| NF5 | Security | All data encrypted at rest & in transit; API key auth |
| NF6 | Cost efficiency | Prefer serverless / scale-to-zero where possible |

---

## 3. Architecture Overview (Diagram Description)

The infrastructure is organized in **five horizontal layers** from bottom to top,
plus a vertical **CI/CD & governance** column on the right side.

> **Tip for the diagram:** Draw five horizontal swim lanes (one per layer) with
> arrows showing data flow from bottom-left (sources) to top-right (consumers).
> Use the CI/CD column on the right as a vertical bar that touches every layer.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CONSUMERS                                    │
│  Trading Dashboard  ·  Grid Operator Portal  ·  Internal BI        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTPS / WebSocket
┌──────────────────────────────▼──────────────────────────────────────┐
│                     5. SERVING LAYER                                │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────────┐  │
│  │ API Gateway / │──▶│ FastAPI      │──▶│ Model (XGBoost         │  │
│  │ Load Balancer │   │ Container(s) │   │ pipeline from registry)│  │
│  └──────────────┘   └──────┬───────┘   └────────────────────────┘  │
│                            │ logs predictions                       │
└────────────────────────────┼────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                     4. MONITORING LAYER                              │
│                                                                     │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────────────┐ │
│  │ Prediction Log│  │ Drift Detector│  │ Alert Manager           │ │
│  │ (JSONL / DB)  │─▶│ PSI + KS Test │─▶│ → Slack / Email / Pager │ │
│  └───────────────┘  └───────────────┘  └─────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────────────────────┐                      │
│  │ Grafana Dashboards                        │                      │
│  │  • Prediction accuracy vs. actuals        │                      │
│  │  • Latency p95/p99                        │                      │
│  │  • Drift scores per feature               │                      │
│  │  • Prediction volume & error distribution │                      │
│  └───────────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     3. TRAINING & EXPERIMENT LAYER                   │
│                                                                     │
│  ┌────────────┐   ┌──────────────┐   ┌──────────────────────────┐  │
│  │  Airflow    │──▶│ Training Job │──▶│ MLflow                   │  │
│  │  Scheduler  │   │ (src/train)  │   │  • Experiment tracking   │  │
│  │  (DAG)      │   │              │   │  • Model registry        │  │
│  └─────┬──────┘   └──────────────┘   │  • Artifact store (S3)   │  │
│        │                              └──────────────────────────┘  │
│        │ triggers on: schedule (weekly) OR drift alert              │
│        │                                                            │
│  ┌─────▼──────────────────────────────────────────────────────────┐ │
│  │  Training Pipeline Steps                                       │ │
│  │  1. Validate input data (schema + stats)                       │ │
│  │  2. Feature engineering (temporal, cyclical, lag, rolling)      │ │
│  │  3. Train XGBoost with early stopping                          │ │
│  │  4. Evaluate on validation set (MAE, RMSE, R², MAPE)          │ │
│  │  5. Compare against current production model (champion)        │ │
│  │  6. If better → promote to "Production" in model registry      │ │
│  │  7. Notify API to hot-reload new model                         │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     2. DATA LAYER                                    │
│                                                                     │
│  ┌──────────────────┐  ┌─────────────────┐  ┌───────────────────┐  │
│  │ Raw Zone          │  │ Processed Zone  │  │ Feature Store     │  │
│  │ (landing area)    │─▶│ (cleaned,       │─▶│ (precomputed      │  │
│  │                   │  │  validated)      │  │  features, shared │  │
│  │ • weather.parquet │  │ • train.parquet  │  │  train & serve)   │  │
│  │ • generation.pqt  │  │ • val.parquet    │  │                   │  │
│  │ • metadata.pqt    │  │ • test.parquet   │  │                   │  │
│  └──────────────────┘  └─────────────────┘  └───────────────────┘  │
│                                                                     │
│  Storage: Cloud Object Store (S3 / Azure Blob / GCS)               │
│  Catalog: Data versioning (DVC or Delta Lake)                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     1. INGESTION LAYER                               │
│                                                                     │
│  ┌─────────────────┐     ┌──────────────────┐                      │
│  │ Weather API      │────▶│ Ingestion        │                      │
│  │ (Open-Meteo /    │     │ Pipeline         │                      │
│  │  KNMI / ECMWF)   │     │ (Airflow task)   │                      │
│  └─────────────────┘     │                  │                      │
│                           │ • Fetch          │                      │
│  ┌─────────────────┐     │ • Validate       │                      │
│  │ SCADA / Inverter │────▶│ • Transform      │──▶ Raw Zone          │
│  │ Telemetry (MQTT  │     │ • Land in store  │                      │
│  │  or REST)         │     │                  │                      │
│  └─────────────────┘     └──────────────────┘                      │
│                                                                     │
│  ┌─────────────────┐                                                │
│  │ Park Metadata    │ (location, capacity, tilt, azimuth)           │
│  │ (manual / CRM)   │                                               │
│  └─────────────────┘                                                │
└─────────────────────────────────────────────────────────────────────┘

  ║                                                                   ║
  ║   CI/CD & GOVERNANCE (vertical column on the right)               ║
  ║                                                                   ║
  ║   • GitHub repo + branch protection                               ║
  ║   • GitHub Actions / GitLab CI                                    ║
  ║     – lint, unit tests, integration tests                         ║
  ║     – build Docker image → push to container registry             ║
  ║     – deploy to staging → run smoke tests → promote to prod       ║
  ║   • Infrastructure as Code (Terraform / Pulumi)                   ║
  ║   • Secret management (Vault / AWS Secrets Manager)               ║
  ║   • Role-based access control (RBAC)                              ║
  ║   • Audit logging                                                 ║
  ║                                                                   ║
```

---

## 4. Layer-by-Layer Details

### 4.1 Ingestion Layer

| Component | Description | Technology Options |
|-----------|-------------|--------------------|
| Weather data source | Hourly NWP forecasts (GFS, ECMWF, Open-Meteo) | REST API polling via Airflow sensor |
| Generation data source | Actual kW output per inverter / park | MQTT broker → Kafka/Pub-Sub, or REST polling |
| Park metadata | Static info: lat, lon, dc_capacity, ac_capacity, tilt, azimuth | Manual upload or CRM sync |
| Ingestion orchestration | Scheduled fetch + validate + land | Airflow DAG (`ingest_weather`, `ingest_generation`) |
| Data validation | Schema checks, range checks, null checks | Great Expectations or Pandera |

**Key design decision:** Weather data is *pull-based* (we call the API on schedule).
Generation data can be *push-based* (inverters stream via MQTT) or pull-based
depending on the customer's SCADA system.

### 4.2 Data Layer

| Component | Description |
|-----------|-------------|
| **Raw Zone** | Immutable landing area. Data stored as-is in Parquet, partitioned by date and park_id. |
| **Processed Zone** | Cleaned, validated, and joined data. This is where train/val/test splits live. |
| **Feature Store** | Precomputed features (lag, rolling, cyclical encodings). Same logic used for training and serving, eliminating **train-serve skew**. |
| **Data versioning** | DVC or Delta Lake to version datasets alongside code. |

**Why a Feature Store matters:**
Your current project duplicates feature engineering in [train.py](src/train.py) and
[app.py](api/app.py). In production this is a risk — any difference causes
*train-serve skew*. A feature store computes features once and serves them to both
training and inference.

### 4.3 Training & Experiment Layer

| Component | Description |
|-----------|-------------|
| **Orchestrator** | Airflow DAG with tasks: validate → feature-eng → train → evaluate → promote |
| **Compute** | Containerized training job (current: local Python, target: Kubernetes Job or cloud ML job) |
| **Experiment tracking** | MLflow: logs params, metrics (MAE, RMSE, R², MAPE), artifacts |
| **Model registry** | MLflow Model Registry with stages: Staging → Production → Archived |
| **Champion/Challenger** | New model must beat current production model on validation set before promotion |

**Retraining triggers:**
1. Scheduled (weekly cron)
2. Drift-triggered (monitoring layer detects PSI > 0.2 on key features)
3. Manual (data scientist triggers via Airflow UI)

### 4.4 Monitoring Layer

This builds directly on your existing [monitoring.py](src/monitoring.py):

| Component | What it monitors | Current | Target |
|-----------|-----------------|---------|--------|
| **Prediction logging** | Every request + response + latency | JSONL files | Structured DB (Postgres) or time-series DB (InfluxDB) |
| **Data drift detection** | PSI & KS test on input features | In code | Scheduled Airflow task, daily |
| **Concept drift detection** | Prediction accuracy vs. actuals | Not yet | Compare forecast vs. metered generation (lagged) |
| **Operational metrics** | Latency, error rate, throughput | Latency in logs | Prometheus + Grafana dashboard |
| **Alerting** | Drift, latency, errors, low volume | Print to console | Slack webhook + PagerDuty for critical |

**Dashboard panels (Grafana):**
1. **Accuracy** — MAE over time, per park
2. **Drift** — PSI heatmap per feature, per day
3. **Operations** — Request count, p95 latency, error rate
4. **Model lifecycle** — Current production model version, last retrain date, next scheduled retrain

### 4.5 Serving Layer

| Component | Description |
|-----------|-------------|
| **API** | FastAPI application (your existing [app.py](api/app.py)) |
| **Containerization** | Docker image (your existing [Dockerfile](docker/Dorckerfile.api)) |
| **Orchestration** | Kubernetes (EKS/AKS/GKE) with Horizontal Pod Autoscaler |
| **Load balancer** | Cloud ALB / Nginx Ingress — routes traffic, terminates TLS |
| **Model loading** | On startup: pull latest "Production" model from MLflow registry |
| **Hot reload** | On new model promotion: rolling restart of API pods (zero downtime) |
| **Auth** | API key authentication via API Gateway |
| **Rate limiting** | Per-client rate limits at API Gateway |

### 4.6 CI/CD & Governance

```
  Code push ──▶ Lint + Unit Tests ──▶ Build Docker Image ──▶ Push to Registry
                                                                    │
                            ┌───────────────────────────────────────┘
                            ▼
                  Deploy to Staging ──▶ Integration Tests ──▶ Smoke Tests
                                                                    │
                            ┌───────────────────────────────────────┘
                            ▼
                  Manual Approval Gate ──▶ Deploy to Production
```

| Practice | Tool |
|----------|------|
| Version control | Git + GitHub/GitLab |
| CI pipeline | GitHub Actions / GitLab CI |
| Container registry | ECR / ACR / Artifact Registry |
| Infrastructure as Code | Terraform |
| Secrets | AWS Secrets Manager / Azure Key Vault |
| Access control | RBAC on cloud + Kubernetes namespaces |

---

## 5. Cloud Service Mapping

| Component | AWS | Azure | GCP |
|-----------|-----|-------|-----|
| Object storage | S3 | Blob Storage | GCS |
| Container orchestration | EKS (Fargate) | AKS | GKE Autopilot |
| Workflow orchestration | MWAA (managed Airflow) | — | Cloud Composer |
| Experiment tracking | SageMaker / self-hosted MLflow | Azure ML | Vertex AI |
| Model registry | SageMaker Model Registry / MLflow | Azure ML | Vertex AI |
| API gateway | API Gateway | API Management | Cloud Endpoints |
| Monitoring | CloudWatch | Application Insights | Cloud Monitoring |
| Dashboards | Managed Grafana | Managed Grafana | Looker / Grafana |
| Alerting | SNS + Lambda | Action Groups | Cloud Alerting |
| Secrets | Secrets Manager | Key Vault | Secret Manager |
| IaC | Terraform / CDK | Terraform / Bicep | Terraform |

---

## 6. Talking Points for the Interview

When presenting this blueprint, emphasize:

1. **Train-serve consistency** — The feature store eliminates skew between training
   and serving. This is a common production ML failure mode.

2. **Champion/Challenger pattern** — New models are never deployed blindly. They must
   outperform the current model on a held-out set.

3. **Observability is not optional** — Drift detection, latency tracking, and
   accuracy monitoring are first-class citizens, not afterthoughts. You already built
   the foundation in your monitoring module.

4. **Separation of concerns** — Each layer has a single responsibility. The API
   doesn't train models. The training pipeline doesn't serve traffic. Airflow
   orchestrates but doesn't compute.

5. **Scalability path** — The architecture supports going from 1 park to 100+ parks
   by parameterizing pipelines on `park_id` and scaling API pods horizontally.

6. **Cost awareness** — Use scale-to-zero (Fargate Spot, GKE Autopilot) for training
   compute that's only needed weekly. Keep always-on costs limited to the API and
   monitoring.
