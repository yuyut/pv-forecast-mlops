# AI Roadmap — PV Solar Forecast Platform

> **Audience:** C-level executives of the energy company
> **Format:** Present this as 3–4 slides maximum. Keep text minimal on slides
> and use this document as your speaker notes.

---

## Executive Summary

We will deliver an AI-powered solar generation forecasting platform in four
phases over ~16 weeks. Each phase ends with a **working deliverable** that
adds measurable business value. The approach is iterative — we de-risk early
by proving the model works before investing in infrastructure.

| Phase | Duration | Headline | Business Value |
|-------|----------|----------|----------------|
| 1. Prove | Weeks 1–4 | Working model + API | "We can forecast" |
| 2. Productionize | Weeks 5–8 | Reliable, monitored pipeline | "We can trust the forecast" |
| 3. Scale | Weeks 9–12 | Multi-park, automated retraining | "Every park gets a forecast" |
| 4. Optimize | Weeks 13–16+ | Better models, new use cases | "The forecast keeps improving" |

---

## Phase 1: Prove (Weeks 1–4)

**Goal:** Demonstrate that an ML model can forecast PV generation accurately
enough to be useful for the business.

### Deliverables
| # | Deliverable | Definition of Done |
|---|-------------|--------------------|
| 1.1 | Exploratory data analysis | Documented data quality report: missing values, distributions, correlations |
| 1.2 | Baseline model (XGBoost) | Trained model with MAE, RMSE, R², MAPE evaluated on a held-out time-based test set |
| 1.3 | REST API for predictions | FastAPI endpoint accepting weather features, returning kW prediction, containerized with Docker |
| 1.4 | Demo to stakeholders | Live demo: call API with real weather data, show prediction vs. actual |

### Key Activities
- Collect and clean historical weather + generation data for a **single pilot park**
- Engineer features: temporal (hour, day-of-year), cyclical encodings, lag features,
  rolling statistics, solar geometry proxies
- Train XGBoost with scikit-learn pipeline (imputation + scaling + regressor)
- Set up MLflow for experiment tracking
- Build FastAPI serving endpoint with health check
- Containerize with Docker

### Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Data quality issues (gaps, wrong units) | Model accuracy | Invest time in EDA; validate ranges; impute conservatively |
| Overfitting to single park | Poor generalization | Use time-based train/val/test split (no data leakage) |
| Weather API reliability | Missing inputs at serving time | Cache last-known forecast; implement fallback (persistence model) |

### Success Criteria
- MAE < 10% of average park capacity
- API responds in < 200 ms (p95)
- Stakeholders agree the forecast is directionally useful

### Status in Current Project
> **This phase is largely complete.** You have a trained XGBoost model, a FastAPI
> endpoint, Docker container, MLflow tracking, and feature engineering. Your focus
> in the interview should be: "Phase 1 is done. Here's what we learned, here's the
> accuracy we achieved, and here's the plan to make it production-ready."

---

## Phase 2: Productionize (Weeks 5–8)

**Goal:** Make the forecast reliable enough that the business can depend on it
daily without manual intervention.

### Deliverables
| # | Deliverable | Definition of Done |
|---|-------------|--------------------|
| 2.1 | Automated data ingestion | Airflow DAGs ingest weather + generation data on schedule |
| 2.2 | Automated training pipeline | Airflow DAG: validate → train → evaluate → promote (champion/challenger) |
| 2.3 | Monitoring & alerting | Drift detection (PSI, KS), latency tracking, Slack/email alerts |
| 2.4 | CI/CD pipeline | Push to main → test → build image → deploy to staging → promote to prod |
| 2.5 | Infrastructure as Code | All infra defined in Terraform; reproducible environments |

### Key Activities
- Extend Airflow DAG from single training task to full pipeline
  (your current [pv_pipeline.py](airflow/dags/pv_pipeline.py) is the starting point)
- Add data validation step (Great Expectations / Pandera) before training
- Implement champion/challenger model comparison in MLflow Model Registry
- Deploy monitoring module (your [monitoring.py](src/monitoring.py)) as a scheduled job
- Build Grafana dashboards: accuracy over time, drift scores, API latency
- Set up CI/CD: GitHub Actions → Docker build → deploy to Kubernetes
- Write integration tests: API contract tests, data pipeline smoke tests
- Define and provision cloud infrastructure with Terraform

### Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Silent model degradation | Bad forecasts erode trust | Monitoring + alerting on accuracy vs. actuals (when metered data arrives) |
| Pipeline failures block retraining | Model grows stale | Alert on DAG failures; document manual retraining runbook |
| Infrastructure costs overrun | Budget pressure | Use spot/preemptible instances for training; scale-to-zero API outside peak |

### Success Criteria
- Zero manual steps from code merge to production deployment
- Drift alert fires within 1 hour of distribution shift
- Retraining pipeline runs end-to-end without human intervention
- 99.5% API uptime over a 2-week measurement period

---

## Phase 3: Scale (Weeks 9–12)

**Goal:** Extend the platform from a single pilot park to the full portfolio.

### Deliverables
| # | Deliverable | Definition of Done |
|---|-------------|--------------------|
| 3.1 | Multi-park support | API + training accept `park_id`; separate model per park or single model with park features |
| 3.2 | Park onboarding process | Documented: add new park metadata → data starts flowing → model trains automatically |
| 3.3 | Feature store | Shared feature computation for training and serving (eliminates train-serve skew) |
| 3.4 | Operational dashboard | Grafana: per-park accuracy, fleet-wide overview, onboarding status |
| 3.5 | Auto-scaling API | Kubernetes HPA: scale pods based on request volume |

### Key Activities
- **Architecture decision: one model per park vs. one global model**
  - Per-park: better accuracy, higher maintenance
  - Global with park features (lat, lon, capacity, tilt): simpler, may underfit
  - **Recommended:** Start with global model + park features; switch to per-park only
    where global accuracy is insufficient
- Parameterize training pipeline on `park_id`
- Implement feature store (Feast or custom) to serve precomputed lag/rolling features
- Add park onboarding: metadata entry triggers automatic inclusion in next training run
- Configure Kubernetes Horizontal Pod Autoscaler for API deployment
- Load test the API (Locust / k6) to validate scaling behavior

### Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Per-park models multiply maintenance burden | Engineering overhead | Start with global model; only split when proven necessary |
| New parks have no history | Cold-start problem | Use global model for new parks; transfer learn from similar parks |
| Feature store adds complexity | Engineering time | Start simple (shared Python module); migrate to Feast only at 50+ parks |

### Success Criteria
- 10+ parks running on the platform
- New park operational within 48 hours of metadata entry
- No train-serve skew (feature parity verified by automated test)

---

## Phase 4: Optimize (Weeks 13–16, then ongoing)

**Goal:** Continuously improve forecast accuracy and expand to new use cases.

### Deliverables
| # | Deliverable | Definition of Done |
|---|-------------|--------------------|
| 4.1 | Model improvements | Evaluate LightGBM, neural networks (temporal fusion transformer), ensemble methods |
| 4.2 | Additional data sources | Satellite imagery, snow cover, soiling index, panel degradation curves |
| 4.3 | Probabilistic forecasts | Quantile regression: provide P10/P50/P90 confidence intervals |
| 4.4 | A/B testing framework | Shadow mode: run new model alongside production, compare on live data before switching |
| 4.5 | New use cases | Battery optimization, energy trading signals, maintenance scheduling |

### Key Activities
- **Model experimentation:**
  - LightGBM / CatBoost as drop-in XGBoost alternatives
  - Temporal Fusion Transformer for longer-horizon forecasts
  - Stacking ensemble combining weather-model-specific sub-models
- **Probabilistic forecasts:** Replace point predictions with quantile regression.
  The trading desk needs confidence intervals (P10–P90) to assess risk.
- **A/B / shadow testing:** Route a percentage of traffic to challenger model;
  compare predictions against actuals without affecting production output.
- **New data sources:**
  - Satellite cloud imagery (Meteosat / GOES) for nowcasting (0–6h)
  - Panel degradation models (age, soiling) as additional features
  - Electricity price signals for joint optimization
- **New use cases:**
  - Battery charge/discharge scheduling using PV forecast + price forecast
  - Predictive maintenance: detect underperforming inverters by comparing
    predicted vs. actual generation

### Success Criteria
- MAE improvement of ≥ 15% over Phase 1 baseline
- Probabilistic forecasts calibrated (P10 covers ~10% of outcomes)
- At least one new use case generating business value

---

## Team & Skills Required

| Phase | Roles Needed | FTE Estimate |
|-------|-------------|-------------|
| 1. Prove | ML Engineer, Data Engineer (part-time) | 1.5 |
| 2. Productionize | ML Engineer, DevOps/MLOps Engineer | 2.0 |
| 3. Scale | ML Engineer, Data Engineer, DevOps Engineer | 2.5 |
| 4. Optimize | ML Engineer (senior), Data Scientist, Domain Expert (energy) | 2.0 |

---

## Cost Estimate (Indicative, Cloud-Based)

| Component | Monthly Cost (est.) | Notes |
|-----------|-------------------|-------|
| Compute (API) | €200–500 | 2–4 pods, scale-to-zero at night |
| Compute (training) | €50–100 | Weekly batch, spot instances |
| Storage (S3/Blob) | €20–50 | Parquet data, model artifacts |
| Managed Airflow | €200–400 | Or self-hosted on K8s for less |
| MLflow hosting | €0–100 | Self-hosted on existing K8s |
| Monitoring (Grafana) | €0–50 | Grafana Cloud free tier or self-hosted |
| **Total** | **~€500–1200/mo** | Scales with number of parks |

> These are ballpark figures. Actual costs depend on cloud provider, region,
> and negotiated pricing. The key message: **this is not expensive infrastructure.**
> The value of a 1% improvement in forecast accuracy often exceeds the
> entire annual platform cost in reduced imbalance charges.

---

## Risk Summary & Mitigations (Cross-Phase)

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | Data quality (gaps, errors, delays) | High | High | Validation gates at ingestion; monitoring; fallback to persistence model |
| R2 | Model accuracy insufficient for business use | Medium | High | Iterative improvement; set clear accuracy targets with stakeholders early |
| R3 | Scope creep (too many parks/features too fast) | Medium | Medium | Phased rollout; prove on pilot before scaling |
| R4 | Key-person dependency | Medium | High | Document everything; pair programming; IaC + CI/CD make the system reproducible |
| R5 | Cloud vendor lock-in | Low | Medium | Use open-source tools (Airflow, MLflow, FastAPI) wherever possible; Terraform abstracts infra |
| R6 | Regulatory/compliance requirements | Low | Medium | Audit logging from Phase 2; encrypted data at rest and in transit |

---

## Summary Timeline

```
Week  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16
      ├──────────────┤
      Phase 1: PROVE
      Model + API + Demo

                     ├──────────────┤
                     Phase 2: PRODUCTIONIZE
                     CI/CD + Monitoring + Automation

                                    ├───────────────┤
                                    Phase 3: SCALE
                                    Multi-park + Feature Store

                                                    ├───────────── ▶
                                                    Phase 4: OPTIMIZE
                                                    Better models + New use cases
                                                    (ongoing)
```

Each phase ends with a **stakeholder review** where we demo progress,
gather feedback, and confirm priorities for the next phase. This ensures
the business stays in control and we build what matters most.

---

## Presenting This Roadmap (Tips)

1. **Lead with business value, not technology.** "In 4 weeks you'll have a working
   forecast. In 8 weeks it will run itself. In 12 weeks every park will have one."

2. **Show the current state.** "We've already completed Phase 1 — here's the model,
   here's the API, here's the accuracy." This builds credibility.

3. **Be honest about unknowns.** "We expect accuracy to improve in Phase 4 but we
   need to validate with real production data first." Executives respect candor.

4. **Have a clear ask.** What do you need from them? Access to data, cloud budget,
   domain expert time, decision on cloud provider.

5. **Keep the slides visual.** One slide per phase. Timeline at the bottom. Bullet
   points for deliverables. No code, no architecture detail (that's for Part 2).
