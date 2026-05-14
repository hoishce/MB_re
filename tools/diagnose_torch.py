#!/usr/bin/env python3
import sys, os, traceback, site

def main():
    print("executable:", sys.executable)
    print("python version:", sys.version)
    print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
    path = os.environ.get("PATH", "")
    print("PATH contains conda:", "conda" in path.lower())
    print("--- sys.path ---")
    for p in sys.path:
        print(p)
    print("--- site.getsitepackages() ---")
    try:
        print(site.getsitepackages())
    except Exception as e:
        print("site.getsitepackages error:", e)
    try:
        import importlib
        t = importlib.import_module("torch")
        print("--- torch ---")
        print("torch version:", getattr(t, "__version__", None))
        print("torch file:", getattr(t, "__file__", None))
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    main()
