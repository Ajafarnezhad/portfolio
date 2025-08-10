# CyberSecuritySuite

[![CI/CD Status](https://github.com/Ajafarnezhad/portfolio/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Ajafarnezhad/portfolio/actions)
[![Code Coverage](https://img.shields.io/badge/coverage-80%25-green)](https://github.com/Ajafarnezhad/portfolio)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Ajafarnezhad/portfolio/blob/main/LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

**CyberSecuritySuite** is a professional-grade cybersecurity tool designed to enhance web application security. Built to comply with **OWASP ASVS v5.0 (2025)**, **NIST SSDF**, and **ISO/IEC 27001:2022**, it offers robust features for port scanning, dependency vulnerability checks, static code analysis, OWASP Top 10 vulnerability detection, and secure encrypted communication. This project showcases enterprise-level software engineering, secure coding practices, and modern deployment techniques, making it an ideal portfolio piece.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Docker Deployment](#docker-deployment)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features
- **Port Scanning**: Identifies open ports on target systems using socket-based scanning.
- **Dependency Scanning**: Integrates Safety to detect vulnerable packages (OWASP A06).
- **Static Code Analysis**: Uses Bandit to identify security issues in code (OWASP A03, A05).
- **OWASP Top 10 Checks**: Manual detection for vulnerabilities like Injection and XSS.
- **Secure Chat**: Real-time, end-to-end encrypted communication using the `cryptography` library.
- **Authentication**: JWT-based secure API access (OWASP A07 compliant).
- **Reporting**: Generates PDF/HTML reports for scan results using ReportLab and Jinja2.
- **Testing**: 80%+ code coverage with pytest.
- **CI/CD**: Automated testing and deployment via GitHub Actions.
- **Containerization**: Dockerized for scalable, consistent deployment.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Ajafarnezhad/portfolio.git
   cd portfolio/CyberSecuritySuite
   ```
2. Install dependencies:
   ```bash
   pip install -r src/requirements.txt
   ```
3. Run the application:
   ```bash
   python src/app.py
   ```
4. Access the web interface at `http://localhost:5000`.

## Docker Deployment
1. Build and run with Docker Compose:
   ```bash
   docker-compose -f docker/docker-compose.yml up -d
   ```
2. Access at `http://localhost:5000`.

## Usage
1. **Login**: Use `/auth/login` with credentials (default: `admin`/`secure_password`).
2. **Scan Ports**: POST to `/scan/port` with JSON payload `{ "target": "192.168.1.1", "ports": "1-1000" }`.
3. **Dependency Check**: POST to `/scan/dependencies` to identify vulnerable packages.
4. **Static Analysis**: POST to `/scan/static` with `{ "file_path": "path/to/code" }`.
5. **Secure Chat**: Connect to `127.0.0.1:8888` using a client with encryption support.
6. View reports in the `reports/` directory (PDF format).

**Note**: Always obtain explicit consent before scanning any system to comply with ethical and legal standards.

## Documentation
- Generate Sphinx documentation:
  ```bash
  sphinx-build -b html docs/ docs/_build/
  ```
- View at `docs/_build/index.html`.
- API specifications are available via Swagger at `/docs`.

## Contributing
Contributions are welcome! Please follow the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md). Key steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit changes (`git commit -m "Add new feature"`).
4. Push and open a Pull Request.

All code must pass PEP 8, Bandit, Safety, and pytest checks.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
- Maintainer: Amirhossein Jafarnezhad
- GitHub: [Ajafarnezhad](https://github.com/Ajafarnezhad)
- Email: [aiamirjd@gamil.com]
- Original Contributor: Sathwik R ([github.com/cicada0007](https://github.com/cicada0007))

---

**Portfolio Note**: This project demonstrates expertise in secure software development, compliance with international standards (OWASP, NIST, ISO), and modern DevOps practices (Docker, CI/CD). Check out the [GitHub repository](https://github.com/Ajafarnezhad/portfolio) for source code, a demo video, and detailed documentation.