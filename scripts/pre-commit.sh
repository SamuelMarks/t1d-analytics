#!/usr/bin/env bash
set -e

echo "Running Pre-commit checks..."

echo "1. Running ruff check..."
ruff check .

echo "2. Running prettier check..."
cd web
npx prettier --check .
cd ..

echo "3. Running python tests and coverage..."
PYTHONPATH=src pytest tests/ --cov=src/t1d_analytics --cov-report=json

echo "4. Running frontend tests..."
cd web
npm run test
cd ..

echo "5. Running e2e tests..."
cd web
npx playwright test
cd ..

echo "6. Generating shields in README.md..."

python3 - << 'PYEOF'
import json
import re
import subprocess
import sys

try:
    with open('coverage.json') as f:
        data = json.load(f)
        test_cov = int(data['totals']['percent_covered'])
except Exception as e:
    print(f"Failed to read coverage.json: {e}")
    sys.exit(1)

test_color = "brightgreen" if test_cov >= 90 else "yellow" if test_cov >= 70 else "red"
test_shield = f"[![Coverage](https://img.shields.io/badge/Coverage-{test_cov}%25-{test_color}.svg)](#)"

try:
    result = subprocess.run(['interrogate', 'src/t1d_analytics'], capture_output=True, text=True)
    match = re.search(r'actual: ([0-9.]+)%', result.stdout)
    doc_cov = int(float(match.group(1))) if match else 0
except Exception as e:
    print(f"Failed to run interrogate: {e}")
    sys.exit(1)

doc_color = "brightgreen" if doc_cov >= 90 else "yellow" if doc_cov >= 70 else "red"
doc_shield = f"[![Docs](https://img.shields.io/badge/Docs-{doc_cov}%25-{doc_color}.svg)](#)"

readme_path = 'README.md'
with open(readme_path, 'r') as f:
    content = f.read()

# Remove existing coverage shields if any
content = re.sub(r'\[!\[Coverage\].*\n?', '', content)
content = re.sub(r'\[!\[Docs\].*\n?', '', content)

# Insert after License shield
license_pattern = r'(\[!\[License\][^\n]+\n)'
replacement = rf'\1{test_shield}\n{doc_shield}\n'

if re.search(license_pattern, content):
    content = re.sub(license_pattern, replacement, content)
else:
    print("License shield not found in README.md")

with open(readme_path, 'w') as f:
    f.write(content)
PYEOF

git add README.md
echo "Pre-commit checks passed successfully!"
