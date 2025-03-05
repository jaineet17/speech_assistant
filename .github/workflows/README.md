# GitHub Actions Workflows

This directory contains the GitHub Actions workflows for the Speech Assistant project.

## CI/CD Pipeline

The main CI/CD pipeline is defined in `ci.yml`. It runs on every push to the `main` branch and on every pull request to the `main` branch.

### Jobs

- **test**: Runs linting and unit tests on the Python codebase.
- **build**: Builds a Docker image for the application.

## Frontend CI

The frontend CI pipeline is defined in `frontend.yml`. It runs on every push to the `main` branch and on every pull request to the `main` branch, but only if files in the `frontend` directory have changed.

### Jobs

- **build**: Builds the frontend application.

## Troubleshooting

If you encounter issues with the CI/CD pipeline, check the following:

1. Make sure all dependencies are correctly specified in `requirements.txt` and `setup.py`.
2. Ensure that all tests are passing locally before pushing.
3. Check that the code follows the style guidelines enforced by flake8 and black.
4. Verify that the Docker build works locally. 