#!/usr/bin/env python3
"""
Debug script to test interrupt import
"""

print("Testing interrupt import...")

try:
    from langgraph.types import interrupt
    print(f"✅ Import successful: {type(interrupt)}")
    print(f"✅ Is callable: {callable(interrupt)}")
    print(f"✅ Function details: {interrupt}")
    print(f"✅ Module: {interrupt.__module__ if hasattr(interrupt, '__module__') else 'No module'}")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()