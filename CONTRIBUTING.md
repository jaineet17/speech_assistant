# Contributing to Speech Assistant

Thank you for your interest in contributing to the Speech Assistant project! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/yourusername/speech-assistant.git
cd speech-assistant
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install development dependencies**

```bash
make setup
# or manually: pip install -e ".[dev]"
```

## Development Workflow

1. **Create a new branch for your feature or bugfix**

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bugfix-name
```

2. **Make your changes and run tests**

```bash
# Run all tests
make test

# Run only unit tests
make test-unit

# Run only integration tests
make test-integration
```

3. **Format and lint your code**

```bash
# Format code
make format

# Run linting
make lint
```

4. **Commit your changes**

```bash
git add .
git commit -m "Your descriptive commit message"
```

5. **Push your changes and create a pull request**

```bash
git push origin feature/your-feature-name
```

Then go to GitHub and create a pull request from your branch.

## Code Style Guidelines

- Follow PEP 8 style guidelines for Python code
- Use docstrings for all functions, classes, and modules
- Write unit tests for new functionality
- Keep functions small and focused on a single task
- Use meaningful variable and function names

## Pull Request Process

1. Ensure your code passes all tests and linting
2. Update documentation if necessary
3. Add your changes to the CHANGELOG.md file
4. Make sure your PR description clearly describes the changes and their purpose
5. Wait for code review and address any feedback

## Adding New Dependencies

If you need to add a new dependency:

1. Add it to `requirements.txt` with a specific version
2. Add it to `setup.py` in the `install_requires` list
3. Document why the dependency is needed in your PR description

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License. 