import yaml
import sys

try:
    with open('dvc.yaml', 'r') as f:
        data = yaml.safe_load(f)
    print("✅ YAML is valid!")
    print("Found stages:", list(data['stages'].keys()))
except Exception as e:
    print("❌ YAML validation failed:", str(e))
    sys.exit(1)
