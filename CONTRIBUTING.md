# Contributing to AI Recommendation Engine

Thank you for your interest! Here's how to contribute effectively.

## Development Setup

```bash
git clone https://github.com/Aranya2801/recommendation-engine.git
cd recommendation-engine
cp .env.example .env  # Add your API keys
docker compose up -d postgres redis qdrant
cd backend && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
uvicorn main:app --reload
```

## PR Guidelines

- One feature/fix per PR
- Include tests for new ML models (`tests/ml/`)
- Follow Black + Ruff for Python, ESLint for TypeScript
- Update CHANGELOG.md with your change
- All CI checks must pass

## Areas We Need Help

- New ML model implementations (Knowledge Graph, RL)
- Dataset integrations (arXiv API, Spotify API)
- Frontend components (notification system, mobile PWA)
- Documentation improvements
- Performance benchmarks

## Code Style

```bash
# Python
black backend/ ml/
ruff check backend/ ml/ --fix

# TypeScript
cd frontend && npm run lint
```

## Reporting Issues

Please include: Python version, OS, error traceback, and steps to reproduce.
