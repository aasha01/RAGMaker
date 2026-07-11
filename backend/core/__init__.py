"""Orchestration layer.

Modules here wire the swappable stages together but never depend on a concrete
strategy — they resolve strategies by key through each stage's REGISTRY.

    recipe.py     build / save / load a Recipe (config + artifacts)   [Step 3]
    inspector.py  per-stage visualisation helpers                     [Step 4/6]
    evaluator.py  query-time comparison across recipes + providers    [Step 7]

The modules themselves are added in later build steps; this package marker
keeps the layout from ARCHITECTURE.md section 9 in place from the start.
"""
