#!/bin/bash

# SUMO Traffic Generator Test Suite Setup Script
# Installs dependencies and validates test environment

set -e

echo "🔧 Setting up SUMO Traffic Generator Test Suite..."

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Recommendation: activate virtual environment first"
    echo "   Example: source .venv/bin/activate"
fi

# Install test dependencies
echo "📦 Installing test dependencies..."
pip3 install pytest pytest-cov pytest-timeout pytest-xdist psutil

# Check SUMO installation
echo "🚗 Checking SUMO installation..."
if command -v sumo &> /dev/null; then
    echo "✅ SUMO found: $(which sumo)"
    sumo --version | head -1
else
    echo "❌ SUMO not found in PATH"
    echo "   Please install SUMO and ensure it's in your PATH"
    echo "   Ubuntu/Debian: sudo apt-get install sumo sumo-tools sumo-doc"
    echo "   macOS: brew install sumo"
    exit 1
fi

# Check required SUMO tools
echo "🔍 Checking SUMO tools..."
required_tools=("netgenerate" "netconvert")
for tool in "${required_tools[@]}"; do
    if command -v "$tool" &> /dev/null; then
        echo "✅ $tool found"
    else
        echo "❌ $tool not found - required for testing"
        exit 1
    fi
done

# Check project structure
echo "📁 Validating project structure..."
required_dirs=("src" "tests" "evaluation")
for dir in "${required_dirs[@]}"; do
    if [[ -d "$dir" ]]; then
        echo "✅ $dir/ directory found"
    else
        echo "❌ $dir/ directory missing"
        exit 1
    fi
done

# Check if CLI module is importable
echo "🐍 Testing Python imports..."
if python3 -c "from src import cli" 2>/dev/null; then
    echo "✅ CLI module importable"
else
    echo "⚠️  CLI module import failed - some tests may fail"
fi

if python3 -c "from src.config import GridConfig" 2>/dev/null; then
    echo "✅ Config module importable"
else
    echo "⚠️  Config module import failed - some tests may fail"
fi

# Check Tree Method sample data (optional)
echo "🌳 Checking Tree Method sample data..."
if [[ -d "evaluation/datasets/networks" ]]; then
    echo "✅ Tree Method sample data found"
else
    echo "⚠️  Tree Method sample data not found"
    echo "   Sample tests will be skipped"
fi

# Run basic test validation
echo "🧪 Running basic test validation..."
if pytest tests/system/test_smoke.py::TestQuickValidation::test_cli_help -v --tb=short; then
    echo "✅ Basic test validation passed"
else
    echo "❌ Basic test validation failed"
    echo "   Check the error above and ensure environment is properly configured"
    exit 1
fi

echo ""
echo "🎉 Test environment setup complete!"
echo ""
echo "📋 Quick Start Commands:"
echo "   pytest tests/ -m smoke -v                    # Run smoke tests"
echo "   pytest tests/ -m scenario -v                 # Run scenario tests"
echo "   pytest tests/ --cov=src --cov-report=html -v # Full test suite with coverage"
echo ""
echo "📖 For detailed instructions, see: tests/TEST_EXECUTION_GUIDE.md"