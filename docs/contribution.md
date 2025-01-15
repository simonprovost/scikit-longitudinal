---
hide:
  - navigation
---

# 🤝 Contributing to Scikit-longitudinal

We appreciate contributions from the community and welcome your ideas, bug reports, and pull requests. Follow this guide to set up the project locally, install dependencies, and contribute effectively.

---

## 🚀 Developer Setup

### Prerequisites
Ensure the following tools are installed:
- [Python 3.9.x](https://www.python.org/downloads/release/python-398/)
- [UV](https://docs.astral.sh/uv/)

---

## 📦 Installing from Source

### Step 1: Clone the Repository
```bash
git clone https://github.com/simonprovost/scikit-longitudinal.git
cd scikit-longitudinal
```

---

### Step 2: Set Up the Environment

#### Using UV (Recommended)
1. **Install and Pin Python Version**:
   ```bash
   uv python install cpython-3.9.21
   uv python pin cpython-3.9.21
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   uv sync --all-groups
   ```

---

### Step 3: Run Tests
To verify your setup:
```bash
uv run pytest -sv tests/
```

---

## ❌ Troubleshooting Errors

### General Issues
If you encounter setup errors:
1. **Deactivate Environment**:
   ```bash
   deactivate
   ```

2. **Clear UV Cache**:
   ```bash
   uv cache clean
   ```
   
_Note: Sometimes removing `uv.lock` and `uv lock` again could help._

3. **Reinstall Dependencies**:
   ```bash
   uv sync --all-groups
   ```

4. **Check Logs**: Review logs for detailed error messages and attempt to resolve conflicts manually.

---

### Windows Known Issues

If you are on Windows, you might encounter issues with line endings. Configure Git to handle line endings appropriately:
```bash
git config --global core.autocrlf true
```

Additionally, we recommend using Docker for a seamless setup on Windows systems, as it avoids most compatibility issues.

---

## ⚙️ Building the Distribution Packages

To build the project for distribution:
```bash
uv build
```

This will generate the distribution packages in the `dist/` directory.

---

## 🛠️ Additional Information

- For detailed command descriptions, refer to the `pyproject.toml` file.
- If you encounter persistent issues, report them on the [issue tracker](https://github.com/simonprovost/scikit-longitudinal/issues).

_🎉 Ready to contribute! Happy coding!_