# MLOps Vehicle Insurance Cross-Sell Prediction

An end-to-end MLOps pipeline for predicting whether existing health-insurance customers are interested in vehicle insurance (cross-sell). The project pulls data from MongoDB, validates it against a defined schema, and produces versioned artifacts under a timestamped directory structure.

## Problem Statement

Given demographic and vehicle information about existing insurance policyholders, predict the binary `Response` column — whether a customer would be interested in vehicle insurance. This is a classic cross-sell prediction problem.

## Features Used

| Feature | Type | Description |
|---------|------|-------------|
| Gender | Categorical | Customer gender |
| Age | Numerical | Customer age |
| Driving_License | Numerical | Whether the customer has a driving license (0/1) |
| Region_Code | Numerical | Region code of the customer |
| Previously_Insured | Numerical | Whether the customer already has vehicle insurance (0/1) |
| Vehicle_Age | Categorical | Age of the vehicle |
| Vehicle_Damage | Categorical | Whether the vehicle was damaged in the past |
| Annual_Premium | Numerical | Premium amount paid by the customer |
| Policy_Sales_Channel | Numerical | Channel used to reach the customer |
| Vintage | Numerical | Number of days the customer has been associated |
| **Response** | **Target** | **1 = interested, 0 = not interested** |

## Project Structure

```
mlops-vehicle-insurance/
├── config/
│   └── schema.yaml                 # Column definitions, numerical/categorical groups
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # MongoDB → CSV → train/test split
│   │   └── data_validation.py      # Schema validation (column count + types)
│   ├── configuration/
│   │   └── mongo_db_connection.py  # MongoDB client (singleton)
│   ├── constants/
│   │   └── __init__.py             # All pipeline constants
│   ├── data_access/
│   │   └── proj1_data.py           # Export MongoDB collection to DataFrame
│   ├── entity/
│   │   ├── config_entity.py        # Pipeline & component config dataclasses
│   │   └── artifact_entity.py      # Artifact dataclasses
│   ├── exception/
│   │   └── __init__.py             # Custom exception with traceback
│   ├── logger/
│   │   └── __init__.py             # Rotating file + console logger
│   ├── pipline/
│   │   └── training_pipeline.py    # Orchestrates the full training pipeline
│   └── utils/
│       └── main_utils.py           # YAML I/O, dill save/load, numpy helpers
├── notebook/
│   ├── exp-notebook.ipynb          # EDA and experimentation
│   └── mongoDB_demo.ipynb          # MongoDB connectivity demo
├── artifact/                       # Auto-generated timestamped pipeline outputs
├── logs/                           # Auto-generated rotating log files
├── demo.py                         # Entry point — runs the training pipeline
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
└── pyproject.toml                  # Build configuration
```

## Pipeline Architecture

```
MongoDB (Proj1-Data collection)
    │
    ▼
┌──────────────────┐
│  Data Ingestion   │  Export → feature_store/data.csv → train/test split
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Data Validation  │  Column count check + numerical/categorical column presence
└────────┬─────────┘
         │
         ▼  (planned)
┌──────────────────┐
│ Data Transform.   │  Feature encoding, scaling (not yet implemented)
└────────┬─────────┘
         │
         ▼  (planned)
┌──────────────────┐
│  Model Trainer    │  Random Forest with hyperparams (not yet implemented)
└────────┬─────────┘
         │
         ▼  (planned)
┌──────────────────┐
│ Model Evaluation  │  Compare against threshold (not yet implemented)
└────────┬─────────┘
         │
         ▼  (planned)
┌──────────────────┐
│  Model Pusher     │  Push to S3 model registry (not yet implemented)
└──────────────────┘
```

## Artifacts

Each pipeline run creates a timestamped directory under `artifact/`:

```
artifact/MM_DD_YYYY_HH_MM_SS/
├── data_ingestion/
│   ├── feature_store/
│   │   └── data.csv            # Full dataset from MongoDB
│   └── ingested/
│       ├── train.csv           # 75% training split
│       └── test.csv            # 25% test split
└── data_validation/
    └── report.yaml             # Validation report (JSON format)
```

## Getting Started

### Prerequisites

- Python 3.11+
- MongoDB instance with the insurance dataset loaded
- (Optional) AWS credentials for S3 model push

### Installation

```bash
# Clone the repository
git clone https://github.com/Xohail69/mlops-vehicle-insurance.git
cd mlops-vehicle-insurance

# Create and activate virtual environment
python -m venv mlops
source mlops/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Set the following before running the pipeline:

```bash
export MONGODB_URL="mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority"
```

| Variable | Required | Description |
|----------|----------|-------------|
| `MONGODB_URL` | Yes | MongoDB connection string (with TLS via certifi) |
| `AWS_ACCESS_KEY_ID` | No | AWS key for S3 model push (future) |
| `AWS_SECRET_ACCESS_KEY` | No | AWS secret for S3 model push (future) |

### Run the Pipeline

```bash
python demo.py
```

This will:
1. Connect to MongoDB and export the `Proj1-Data` collection
2. Save the full dataset to the feature store
3. Split into 75/25 train/test sets
4. Validate both sets against `config/schema.yaml`
5. Write a validation report

### Check Logs

Logs are written to `logs/` with rotating file handlers (5 MB per file, 3 backups):

```bash
ls logs/
```

## Configuration

### `config/schema.yaml`

Defines the expected data schema used by the validation step:

- **columns** — Full column list with expected dtypes
- **numerical_columns** — Columns expected to be numeric
- **categorical_columns** — Columns expected to be categorical
- **drop_columns** — Columns to drop during transformation
- **num_features** / **mm_columns** — Columns for specific transformations (future)

### Key Constants (`src/constants/__init__.py`)

| Constant | Value | Description |
|----------|-------|-------------|
| `DATABASE_NAME` | `Proj1` | MongoDB database name |
| `COLLECTION_NAME` | `Proj1-Data` | MongoDB collection name |
| `TARGET_COLUMN` | `Response` | Target variable |
| `DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO` | `0.25` | Test set ratio |
| `MODEL_TRAINER_EXPECTED_SCORE` | `0.6` | Minimum acceptable model score |
| `APP_HOST` / `APP_PORT` | `0.0.0.0:5000` | Planned API server config |

## Tech Stack

- **Data Storage**: MongoDB Atlas
- **ML Framework**: scikit-learn, imbalanced-learn
- **Serialization**: dill, numpy
- **Configuration**: PyYAML
- **Logging**: Python logging with RotatingFileHandler
- **API** (planned): FastAPI + Uvicorn
- **Cloud** (planned): AWS S3 (boto3)
- **Visualization**: matplotlib, seaborn, plotly

## Roadmap

- [x] Data Ingestion (MongoDB → train/test CSV)
- [x] Data Validation (schema checks)
- [ ] Data Transformation (encoding, scaling)
- [ ] Model Training (Random Forest with hyperparameter tuning)
- [ ] Model Evaluation (threshold-based acceptance)
- [ ] Model Pusher (S3 registry)
- [ ] Prediction API (FastAPI endpoint)
- [ ] Dockerization
- [ ] CI/CD pipeline

## Author

**Sohail Rayeen** — [sohailrayeen0786@gmail.com](mailto:sohailrayeen0786@gmail.com)

## License

This project is for educational and demonstration purposes.
