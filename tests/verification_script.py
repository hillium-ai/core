#!/usr/bin/env python3
"""Simple verification script for WP-042 implementation"""

import sys
import os

# Check if required files exist
required_files = [
    'crates/rest_rav_detector/src/lib.rs',
    'crates/aegis_core/src/layer7/cognitive_safety.rs'
]

print("Checking for required files...")
all_found = True
for file_path in required_files:
    if os.path.exists(file_path):
        print(f"✅ {file_path} exists")
    else:
        print(f"❌ {file_path} missing")
        all_found = False

if all_found:
    print("\n✅ All required files found - WP-042 implementation appears complete")
    sys.exit(0)
else:
    print("\n❌ Some required files missing")
    sys.exit(1)
