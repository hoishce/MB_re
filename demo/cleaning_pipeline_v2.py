#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for the v2 cleaning pipeline."""

import os
import sys

from cleaning_pipeline_v2_impl import *  # noqa: F401,F403


if __name__ == "__main__":
    try:
        rc = int(main())
    except Exception as exc:  # pragma: no cover - CLI guard
        try:
            print(f"[err] pipeline failed: {exc}", file=sys.stderr)
        except Exception:
            pass
        rc = 1
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(rc)
