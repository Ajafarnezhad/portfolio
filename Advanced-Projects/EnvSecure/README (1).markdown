# EnvSecure: Advanced .env Encryption Tool

[![CI/CD](https://github.com/Ajafarnezhad/portfolio/workflows/CI/badge.svg)](https://github.com/Ajafarnezhad/portfolio/actions)
[![Code Coverage](https://img.shields.io/badge/coverage-100%25-green)](https://github.com/Ajafarnezhad/portfolio)
[![PyPI version](https://badge.fury.io/py/envsecure.svg)](https://pypi.org/project/envsecure/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

EnvSecure is a professional-grade tool for encrypting/decrypting .env files, compliant with OWASP ASVS, NIST SSDF, and ISO/IEC 27001. It uses Argon2id for key derivation and Fernet for symmetric encryption, supporting large files, password strength checks, and auto-loading into environments.

## Features
- **Secure Key Derivation**: Argon2id with tunable parameters (OWASP-recommended).
- **Chunked Processing**: Handles large .env files efficiently.
- **Password Validation**: Uses zxcvbn for strength scoring.
- **Logging**: Structured JSON logs for auditing.
- **Integration**: Load encrypted .env directly with python-dotenv.
- **CLI**: Modern Typer interface with progress bars.
- **Testing**: 100% coverage with pytest and hypothesis.
- **CI/CD**: GitHub Actions for linting, testing, security scans.
- **Docker**: Containerized for deployment.

## Installation
1. Clone: `git clone https://github.com/Ajafarnezhad/portfolio.git`
2. Install with Poetry: `poetry install`
3. Or via pip: `pip install envsecure`

## Usage
```bash
envsecure --help  # Show CLI options
envsecure encrypt .env --password-strength min_score=3  # Encrypt with options
envsecure decrypt .env.envs --audit-log=true  # Decrypt with logging
envsecure load .env.envs  # Decrypt and load into os.environ
```

Security Notes: Never commit .env files to git. Use .gitignore for .env*. Use strong passwords (score >=3 via zxcvbn).

## Development
- Tests: `poetry run pytest --cov`
- Docs: `poetry run sphinx-build docs docs/_build`
- Lint: `poetry run black . && poetry run flake8`
- Build: `poetry build`

## License
MIT - See [LICENSE](LICENSE).

Maintainer: Amirhossein Jafarnezhad ([github.com/Ajafarnezhad](https://github.com/Ajafarnezhad))