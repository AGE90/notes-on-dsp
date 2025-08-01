#!/bin/bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if a command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_message "$RED" "❌ Error: $1 is not installed"
        exit 1
    fi
}

# Check prerequisites
print_message "$YELLOW" "🔍 Checking prerequisites..."
check_command "poetry"
check_command "git"
check_command "python"

# Create necessary directories
print_message "$YELLOW" "📁 Creating project directories..."
mkdir -p data/{raw,processed}
mkdir -p reports/{figures,tables}
mkdir -p notebooks
mkdir -p tests

# Install dependencies
print_message "$YELLOW" "🔧 Installing dependencies..."
poetry install --with dev,test,data-science,viz,notebook

# Set up pre-commit hooks
print_message "$YELLOW" "🔧 Setting up pre-commit hooks..."
poetry run pre-commit install

# Copy environment files
print_message "$YELLOW" "📁 Setting up environment files..."
if [ -f .env.example ]; then
    if [ ! -f .env ]; then
        cp .env.example .env
        print_message "$GREEN" "✅ Created .env file from .env.example"
    else
        print_message "$YELLOW" "⚠️  .env file already exists, skipping"
    fi
else
    print_message "$YELLOW" "⚠️  No .env.example file found, skipping"
fi

# Initialize git if not already initialized
if [ ! -d .git ]; then
    print_message "$YELLOW" "🔧 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit"
    print_message "$GREEN" "✅ Git repository initialized"
fi

# Initialize DVC if not already initialized
if [ ! -d .dvc ]; then
    print_message "$YELLOW" "🔧 Initializing DVC..."
    poetry run dvc init
    print_message "$GREEN" "✅ DVC initialized"
fi

# Create initial .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    print_message "$YELLOW" "📝 Creating .gitignore file..."
    cat > .gitignore << EOL
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
.env
.venv
env/
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Jupyter Notebook
.ipynb_checkpoints

# Testing
.coverage
htmlcov/
.pytest_cache/

# Distribution
dist/
build/

# DVC
.dvc/tmp
.dvc/cache

# Project specific
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep
reports/figures/*
reports/tables/*
!reports/figures/.gitkeep
!reports/tables/.gitkeep
EOL
    print_message "$GREEN" "✅ Created .gitignore file"
fi

# Create .gitkeep files to preserve empty directories
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch reports/figures/.gitkeep
touch reports/tables/.gitkeep

print_message "$GREEN" "✅ Setup complete! You can now start developing."
print_message "$YELLOW" "📝 Next steps:"
echo "1. Review and update .env file if needed"
echo "2. Run 'poetry shell' to activate the virtual environment"
echo "3. Run 'make test' to verify the setup"