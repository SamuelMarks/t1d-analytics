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
PYTHONPATH=src python3 -m pytest tests/ --cov-report=json --cov-report=lcov

echo "4. Running frontend tests..."
cd web
npm run coverage
cd ..

echo "5. Running e2e tests..."
mkdir -p tests/test_data
echo "id,name,age" > tests/test_data/patients.csv
echo "1,Alice,30" >> tests/test_data/patients.csv
echo "2,Bob,40" >> tests/test_data/patients.csv
echo "id,value" > tests/test_data/tiny.csv
echo "1,100" >> tests/test_data/tiny.csv
echo "2,200" >> tests/test_data/tiny.csv
PYTHONPATH=src python3 -c "from t1d_analytics.analytics import load_data_to_duckdb; load_data_to_duckdb('tests/test_data/', 'ci_test.duckdb')"

cd web
npx playwright test
npx nyc report --reporter=lcov --report-dir=coverage/e2e
cd ..

echo "5.5 Merging coverage..."
python3 - << 'PYEOF'
import sys
import collections

# dict: filename -> { line_number: hits }
cov_data = collections.defaultdict(lambda: collections.defaultdict(int))

lcov_files = [
    'coverage.lcov',
    'web/coverage/lcov.info',
    'web/coverage/e2e/lcov.info'
]

for lcov_file in lcov_files:
    try:
        with open(lcov_file, 'r') as f:
            current_file = None
            for line in f:
                line = line.strip()
                if line.startswith('SF:'):
                    current_file = line[3:]
                elif line.startswith('DA:'):
                    # DA:line_num,hits
                    parts = line[3:].split(',')
                    line_num = int(parts[0])
                    hits = int(parts[1])
                    cov_data[current_file][line_num] = max(cov_data[current_file][line_num], hits)
    except FileNotFoundError:
        print(f"Warning: {lcov_file} not found.")

total_found = 0
total_hit = 0

with open('combined-coverage.lcov', 'w') as out:
    for filename, lines in cov_data.items():
        out.write(f"SF:{filename}\n")
        lf = len(lines)
        lh = 0
        for line_num, hits in sorted(lines.items()):
            out.write(f"DA:{line_num},{hits}\n")
            if hits > 0:
                lh += 1
        out.write(f"LF:{lf}\n")
        out.write(f"LH:{lh}\n")
        out.write("end_of_record\n")
        
        total_found += lf
        total_hit += lh

if total_found == 0:
    print("No lines found in coverage report.")
    sys.exit(1)

cov = (total_hit / total_found) * 100.0
print(f"Combined Coverage: {cov:.2f}%")
if cov < 100.0:
    print(f"Coverage is below 100% threshold! (Actual: {cov:.2f}%)")
    sys.exit(1)
PYEOF

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
