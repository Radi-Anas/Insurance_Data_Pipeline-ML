# Insurance Claims Fraud Detection System

A production-grade data pipeline that detects fraudulent insurance claims using machine learning. Built with modern data engineering practices and deployed as a complete ML platform.

---

## The Problem

Insurance fraud costs companies billions annually. When a claim comes in, investigators must manually review each one - taking hours per claim. This system helps prioritize which claims need immediate attention by scoring them for fraud probability.

**Business Impact:**
- Reduce manual review time by 70%
- Catch more fraud with consistent scoring
- Enable data-driven investigation decisions

---

## Architecture Overview

```
1. RAW DATA (CSV) - 10,000 synthetic claims
   └── Insurance claims with customer, policy, incident details

2. ETL PIPELINE (src/data/ingestion/)
   ├── Extract: Load from CSV
   ├── Transform: Clean data, handle missing values, create features
   ├── Validate: Great Expectations, schema registry
   └── Load: Insert into PostgreSQL with connection pooling

3. DATABASE (PostgreSQL)
   └── 10,000 claims, indexed on fraud flag, severity, vehicle make
   └── Connection pooling (5 connections, 10 overflow)

4. ML MODEL (src/models/)
   ├── Features: 36 columns + 14 engineered features
   ├── Target: is_fraud (binary)
   └── Training: XGBoost + RandomForest ensemble

5. API (src/api/)
   ├── /predict: Single claim prediction
   ├── /stats: Aggregated fraud statistics
   ├── /health: System health check
   └── Rate limiting: 10 requests/minute

6. DASHBOARD (src/services/)
   ├── Overview: Fraud rate, severity charts
   ├── Claims browser with filters
   └── Model performance metrics
```

---

## Model Performance

Trained on 10,000 synthetic claims with learnable fraud patterns.

| Metric | Value | Why It Matters |
|--------|-------|----------------|
| Accuracy | 91.7% | Overall correctness |
| AUC-ROC | 0.958 | Excellent ranking ability |
| Precision | 85%+ | When we say fraud, it usually is |
| Recall | 80%+ | We catch most actual fraud |

### Feature Engineering

Created 14 domain-specific features based on fraud detection logic:

| Feature | Formula | Why It Predicts Fraud |
|---------|---------|----------------------|
| `no_witness_injury` | bodily_injuries > 0 AND witnesses = 0 | Injuries without witnesses are suspicious |
| `claim_to_premium_ratio` | total_claim_amount / policy_annual_premium | High claim relative to premium = higher risk |
| `vehicle_property_ratio` | vehicle_claim / property_claim | Unusual damage patterns |
| `injury_ratio` | injury_claim / total_claim_amount | High injury portion may indicate exaggeration |
| `tenure_age_ratio` | months_as_customer / (age * 12) | New customer with old age = suspicious |
| `complex_no_witness` | vehicles > 1 AND witnesses = 0 | Multi-vehicle accidents without witnesses |
| `deductible_claim_ratio` | policy_deductable / total_claim_amount | Low deductible vs high claim |
| `net_capital` | capital-gains - capital-loss | Financial stress indicator |

---

## Tech Stack

| Component | Technology |
|-----------|-------------|
| Language | Python 3.10 |
| Database | PostgreSQL |
| ML | scikit-learn, XGBoost, SHAP |
| API | FastAPI |
| Dashboard | Streamlit |
| Scheduling | Prefect |
| Data Quality | Great Expectations |
| Transformations | dbt |
| Versioning | DVC |
| Testing | pytest |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud | AWS ECS, Kubernetes |

---

## Project Structure

```
.
├── src/
│   ├── api/
│   │   └── app.py              # FastAPI with 8+ endpoints
│   ├── data/
│   │   ├── ingestion/
│   │   │   ├── claims_etl.py   # ETL pipeline
│   │   │   └── synthetic_data.py # Data generator (10K rows)
│   │   ├── processing/
│   │   │   └── transformations/ # dbt models
│   │   └── validation/
│   │       └── data_quality/   # Great Expectations
│   ├── models/
│   │   ├── fraud_model.py      # ML model + SHAP
│   │   ├── fraud_model.pkl
│   │   └── label_encoders.pkl
│   ├── pipelines/
│   │   ├── scheduler.py        # Prefect orchestration
│   │   ├── incremental_etl.py  # Watermark-based processing
│   │   ├── schema_registry.py # Avro schemas
│   │   ├── drift_detection.py # Data/concept drift monitoring
│   │   ├── feature_store.py   # Redis-backed feature store
│   │   └── lineage.py         # Data lineage tracking
│   └── services/
│       └── dashboard.py        # Streamlit dashboard
├── configs/
│   ├── settings.py
│   └── params.yaml
├── tests/                      # 56+ unit tests
├── migrations/                  # Alembic DB migrations
├── scripts/                     # Backup/restore utilities
├── k8s/                        # Kubernetes manifests
├── aws/                        # AWS ECS deployment
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- PostgreSQL (or use Docker)

### Setup

```bash
# Clone and install
git clone https://github.com/Radi-Anas/Insurance_Data_Pipeline-ML.git
cd morocco_re_pipeline
pip install -r requirements.txt

# Configure database
cp .env.development .env
# Edit .env with your DB credentials

# Run ETL and start services
python main.py
```

### Access
- Dashboard: http://localhost:8501
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Database & model status |
| `/predict` | POST | Score a claim (10/min limit) |
| `/predict/batch` | POST | Batch score claims (5/min limit) |
| `/stats` | GET | Fraud statistics (cached) |
| `/claims` | GET | List claims with filters |
| `/features/{policy}` | GET | Get cached features |
| `/model/metrics` | GET | Model performance (API key required) |
| `/model/train` | POST | Retrain model (API key required) |

### Example Usage

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "months_as_customer": 12,
    "age": 35,
    "policy_state": "OH",
    "policy_csl": "250/500",
    "policy_annual_premium": 1200,
    "incident_type": "Single Vehicle Collision",
    "incident_severity": "Major Damage",
    "total_claim_amount": 5000,
    "auto_make": "Toyota",
    "witnesses": 2,
    "bodily_injuries": 0
})

print(response.json())
# {"prediction": 0, "fraud_probability": 0.14, "confidence": "high", "risk_level": "LOW"}
```

### High Fraud Risk Example

```python
response = requests.post("http://localhost:8000/predict", json={
    "months_as_customer": 2,
    "age": 25,
    "policy_state": "NY",
    "policy_csl": "100/300",
    "policy_annual_premium": 800,
    "incident_type": "Multi-Vehicle Collision",
    "incident_severity": "Major Damage",
    "total_claim_amount": 15000,
    "auto_make": "BMW",
    "witnesses": 0,
    "bodily_injuries": 2
})

# {"prediction": 1, "fraud_probability": 0.85, "confidence": "high", "risk_level": "HIGH"}
```

---

## Data Engineering Features

### ETL Pipeline
- Incremental processing with watermark-based approach
- Connection pooling (5 connections, 10 overflow)
- Database indexes for performance
- Synthetic data generation (10,000+ rows)

### Data Quality
- Great Expectations for validation rules
- Schema registry with Avro for data contracts
- Data lineage tracking
- Automated backup/restore scripts

### Transformations
- dbt for SQL-based transformations
- Staging views for clean data
- Mart tables for fraud analytics

### Feature Store
- Redis-backed feature caching
- Precomputed engineered features
- TTL-based invalidation

### Drift Detection
- Population Stability Index (PSI)
- Kolmogorov-Smirnov tests
- Concept drift monitoring
- Alert recommendations

---

## Machine Learning Features

- Ensemble model (XGBoost + RandomForest)
- 14 engineered features based on fraud patterns
- Optimized threshold (0.35) for better recall
- SHAP explainability integration
- Model persistence with joblib
- Feature importance analysis
- Decision logging

### Synthetic Data Generation

Generate realistic insurance claims data with learnable fraud patterns:

```python
from src.data.ingestion.synthetic_data import generate_claims_data

df = generate_claims_data(10000, fraud_rate=0.24)
df.to_csv('data/raw/insurance_claims.csv', index=False)
```

Fraud patterns injected:
- No witness + bodily injuries (60% of fraud)
- High claim-to-premium ratio (70% of fraud)
- New customers (50% of fraud)
- Nighttime incidents (40% of fraud)

---

## Automation & Monitoring

### Prefect Pipeline
- Daily ETL at 2 AM
- Weekly model retraining on Sundays
- Health checks after each run

### Monitoring
- Prometheus metrics endpoint
- Health check endpoints
- Database connection monitoring

### Database
- Alembic migrations
- Connection pooling
- Automated PostgreSQL backups

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

**56+ tests passing**

---

## Deployment

### Docker

```bash
# Start all services
docker-compose up -d

# Services:
# - postgres:5432
# - api:8000
# - dashboard:8501
```

### Kubernetes

```bash
# Apply manifests
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods -l app=fraud-detection
```

### AWS ECS

See `aws/ecs-deployment.md` for:
- ECR image push instructions
- ECS task definition
- Fargate deployment
- RDS/ElastiCache setup

---

## Environment Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `ENV` | Environment (development/staging/production) | development |
| `DATABASE_URL` | PostgreSQL connection string | postgresql://... |
| `API_KEY` | API key for protected endpoints | (none) |
| `LOG_LEVEL` | Logging level (DEBUG/INFO/WARNING) | INFO |

---

## License

MIT
