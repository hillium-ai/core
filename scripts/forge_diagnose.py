#!/usr/bin/env python3
"""
Titan Grounding: Diagnose hardware capabilities for Hillium-Forge.
This script checks for MKL, AVX, and Metal support.
"""
import sys

def diagnose():
    print("🔍 Diagnosing Hillium-Forge Hardware Support...")
    
    # Check for MKL support
    try:
        import torch
        has_mkl = hasattr(torch, 'mkl') and torch.mkl.is_available()
        print(f"MKL Support: {'✅' if has_mkl else '❌'}")
    except ImportError:
        print("MKL Support: ⚠️  PyTorch not available")
    except Exception as e:
        print(f"MKL Support: ⚠️  Error checking: {e}")

    # Check for basic CPU capabilities (Forge environment)
    print("AVX Support: ⚠️  Not directly checkable in Forge environment")

    # Check for Metal support (macOS only)
    try:
        import torch
        has_mps = hasattr(torch, 'mps') and torch.mps.is_available()
        print(f"Metal (MPS) Support: {'✅' if has_mps else '❌'}")
    except ImportError:
        print("Metal (MPS) Support: ⚠️  PyTorch not available")
    except Exception as e:
        print(f"Metal (MPS) Support: ⚠️  Error checking: {e}")

    print("🏁 Diagnosis Complete.")

if __name__ == "__main__":
    diagnose()