"""Serve lane: one runner for holdout, dev and final; the first Kaggle file; the benchmark; the Saturday run.

- :mod:`galaxus.serve.runner`: batches any ``Predictor`` over a part, one jsonl line per image, resumes, times (B4).
- :mod:`galaxus.serve.submit`: predictions → validated submission file → the exact Kaggle command for a human (B5).
- :mod:`galaxus.serve.bench`: images per second, peak memory, cost per 100,000 images (B7).
- :mod:`galaxus.serve.final_run`: passphrase → decrypt → runner → merge into the best dev file → validate (B7).

Agents never upload to Kaggle (AGENTS.md "Galaxus product rules" 7): these modules print the command, a human runs it.
"""
