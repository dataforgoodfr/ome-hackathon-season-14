# OME Hackathon Season 14 - Les puissants gardes forestiers

Data For Good x L'Observatoire des Médias sur l'Écologie

## 🚀 Quick Start

### Prerequisites

- Python 3.11+ (Python 3.14 has limited ML library support)
- Docker & Docker Compose (for inference service)
- Git

### Automated Setup

Run the setup script to install all dependencies:

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Install `uv` (fast Python package manager)
- Install `ty` (type checker)
- Install `ruff` (linter & formatter)
- Create a virtual environment
- Install all project dependencies

### Manual Setup

If you prefer manual setup:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ty (optional, for type checking)
curl -LsSf https://astral.sh/ty/install.sh | sh

# Install ruff (optional, for linting and formatting)
curl -LsSf https://astral.sh/ruff/install.sh | sh

# Sync project dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

## 📊 Dataset Exploration

Explore the hackathon dataset:

```bash
uv run python dataset/explore.py
```

**Dataset Overview:**
- 8,349 text samples from French TV channels
- 4 channels: France 2, TF1, France 3-IDF, M6
- Categories: agriculture/food, mobility/transport, energy, other
- Text types: segments and reports
- Average text length: ~31,874 characters

## 🏗️ Project Structure

```
.
├── dataset/
│   └── explore.py          # Dataset exploration script
├── inference/
│   ├── database_connection.py  # PostgreSQL connection
│   ├── models.py           # Database models
│   ├── predict.py          # Inference logic
│   ├── logs.py             # Logging utilities
│   └── Dockerfile          # Inference service container
├── train/
│   ├── train.py            # Model training script
│   └── Dockerfile          # Training service container
├── models/                 # Trained models directory
├── docker-compose.yml      # Multi-service orchestration
├── pyproject.toml          # Python dependencies
└── setup.sh                # Automated setup script
```

## 🐳 Docker Services

Start the inference service with database and Metabase:

```bash
docker compose up --build inference
```

Available services:
- **inference**: Main prediction service
- **postgres**: PostgreSQL database
- **metabase**: Data visualization dashboard
- **train**: Model training service (optional)

Access Metabase at: http://localhost:3000

## 📦 Installed Dependencies

### Core Dependencies
- **pandas** (3.0.0) - Data manipulation
- **pyarrow** (23.0.0) - Parquet file support
- **huggingface-hub** (1.3.3) - Dataset loading
- **datasets** (4.5.0) - Hugging Face datasets library

### Database
- **sqlalchemy** (2.0.46) - Database ORM
- **psycopg2-binary** (2.9.11) - PostgreSQL driver

### Machine Learning
- **scikit-learn** (1.8.0) - ML utilities and metrics
- **scipy** (1.17.0) - Scientific computing

### ⚠️ Known Issues

**ML Libraries on Python 3.14:**
- `setfit`, `transformers`, and `tokenizers` fail to build due to Rust compilation errors
- **Recommended**: Use Python 3.11 or 3.12 for full ML functionality
- **Alternative**: Use Docker containers which have compatible Python versions

## 🎯 Hackathon Tasks

### Task 1: Text Classification
Classify texts into "report" vs "segment" categories with high efficiency and low resource usage.

**Approach:**
- Use embeddings to analyze semantic structure
- Replicate LLM results with fewer resources
- Focus on FOSS tools and frugal solutions

### Task 2: Category Classification
Classify content into: Agriculture/Food, Mobility/Transport, Energy, or Other.

**Approach:**
- Start with simple solutions (keyword-based)
## 🔧 Development

### Linting and Formatting

Lint your code with `ruff`:

```bash
ruff check .
```

Auto-format your code:

```bash
ruff format .
```

### Type Checking

Check for type errors using `ty`:
### Type Checking

Check for type errors using `ty`:

```bash
ty check
```

### Running Python Scripts

Use `uv run` to execute scripts in the project environment:

```bash
uv run python <script.py>
```

### Adding Dependencies

Add new packages with uv:

```bash
uv add <package-name>
```

## 📝 Evaluation Criteria

1. **Depth of analysis** - Comprehensive solution to the problem
2. **Technical maturity** - Production-ready, containerized solution
3. **Frugality** - Efficient use of resources
4. **FOSS tools** - Prefer open-source over Big Tech models

## 🤝 Contributing

This is a hackathon project for Data For Good x OME Season 14.

**Team:** Les puissants gardes forestiers

## 📄 License

See LICENSE file for details.

## 🔗 Resources

- [uv documentation](https://docs.astral.sh/uv/)
- [ty documentation](https://docs.astral.sh/ty/)
- [OME Project Details](./README-subject.md)
