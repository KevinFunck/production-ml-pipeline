# Contribution Guidelines

## Welcome Contributors! 👋

We appreciate your interest in contributing to the Production ML Pipeline project. This guide will help you get started.

## Code of Conduct

Please be respectful and professional in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch** from `develop`:
   ```bash
   git checkout -b feat/your-feature-name develop
   ```

## Development Workflow

### 1. Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt
```

### 2. Make Your Changes

- Follow PEP 8 style guide
- Add docstrings to all functions
- Write unit tests for new features
- Keep commits atomic and well-documented

### 3. Testing & Quality Checks

```bash
# Run tests
pytest tests/ -v

# Run linting
flake8 src/ tests/

# Format code
black src/ tests/
isort src/ tests/
```

### 4. Commit & Push

Use conventional commit messages:
```
feat: add new feature
fix: fix bug
docs: update documentation
test: add tests
refactor: refactor code
ci: update CI configuration
```

Example:
```bash
git add .
git commit -m "feat: add model versioning support"
git push origin feat/your-feature-name
```

## Pull Request Process

1. **Update documentation** if needed
2. **Ensure all tests pass**:
   ```bash
   pytest tests/ -v --cov=src
   ```
3. **Create Pull Request** on GitHub
4. **Link related issues** in PR description
5. **Describe your changes** clearly
6. **Wait for review** from maintainers

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Related Issues
Closes #(issue number)

## Testing
- [ ] Tests added
- [ ] All tests pass
- [ ] Code coverage maintained

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No new warnings generated
```

## Reporting Issues

### Bug Reports

Please include:
- **Description**: What's the problem?
- **Steps to reproduce**: How to replicate?
- **Expected behavior**: What should happen?
- **Actual behavior**: What happens instead?
- **Environment**: Python version, OS, etc.

### Feature Requests

Please include:
- **Description**: What feature would you like?
- **Use case**: Why is this needed?
- **Examples**: Any relevant examples?

## Areas for Contribution

### High Priority
- [ ] Improve documentation
- [ ] Add more test coverage
- [ ] Performance optimization
- [ ] Bug fixes

### Medium Priority
- [ ] New model algorithms
- [ ] Additional data sources
- [ ] API improvements
- [ ] Docker optimization

### Help Needed
- [ ] Documentation translation
- [ ] Tutorial creation
- [ ] Integration examples
- [ ] Deployment guides

## Development Tips

### Running the Pipeline

```bash
# Train model
python train_pipeline.py

# Start API server
make run-api

# Run tests
make test
```

### Debugging

Enable verbose logging:
```bash
export LOG_LEVEL=DEBUG
python train_pipeline.py
```

## Code Style

We follow PEP 8 with these guidelines:

### Naming
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_CASE`

### Documentation
```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief description of function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something is wrong
    """
    pass
```

## Questions?

- Open an issue for questions
- Check existing issues/discussions first
- Be specific and provide context

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing! 🙏
