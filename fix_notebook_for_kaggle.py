#!/usr/bin/env python3
"""
Fix notebook for Kaggle compatibility
"""

import json

# Load the notebook
with open('Comprehensive_KV_Benchmark.ipynb', 'r') as f:
    nb = json.load(f)

# Ensure kernel metadata
nb['metadata']['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}

nb['metadata']['language_info'] = {
    'codemirror_mode': {'name': 'ipython', 'version': 3},
    'file_extension': '.py',
    'mimetype': 'text/x-python',
    'name': 'python',
    'nbconvert_exporter': 'python',
    'pygments_lexer': 'ipython3',
    'version': '3.10.12'
}

nb['nbformat'] = 4
nb['nbformat_minor'] = 4

# Save
with open('Kaggle_Ready_Comprehensive_Benchmark.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("✓ Created: Kaggle_Ready_Comprehensive_Benchmark.ipynb")
print(f"✓ Cells: {len(nb['cells'])}")
print(f"✓ Kernel: python3")
print(f"✓ Ready for Kaggle!")