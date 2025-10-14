# Contributing to StreamSplit

Thank you for your interest in contributing to StreamSplit! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists in the [Issues](https://github.com/yourusername/StreamSplit-AAAI/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System information (OS, Python version, PyTorch version)

### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/StreamSplit-AAAI.git
   cd StreamSplit-AAAI
   ```

3. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install in development mode:
   ```bash
   pip install -e ".[dev,full]"
   ```

### Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards:
   - Follow PEP 8 style guide
   - Add docstrings to functions and classes
   - Include type hints where appropriate
   - Write unit tests for new features

3. Run tests:
   ```bash
   pytest tests/
   ```

4. Commit your changes:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request

## Code Style

- Use 4 spaces for indentation (not tabs)
- Maximum line length: 79 characters
- Use meaningful variable and function names
- Add comments for complex logic
- Keep functions focused and modular

## Testing

All new features should include tests. Run the test suite before submitting:

```bash
pytest tests/ -v --cov=.
```

## Documentation

- Update README.md if adding new features
- Add docstrings to all public functions and classes
- Update configuration files if needed
- Include usage examples for new features

## Pull Request Process

1. Ensure all tests pass
2. Update documentation
3. Describe your changes in the PR description
4. Link related issues
5. Wait for review and address feedback

## Questions?

Feel free to open an issue for any questions about contributing.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
