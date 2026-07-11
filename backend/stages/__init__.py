"""RAG Lab pipeline stages.

Each stage lives in its own sub-package and follows the same shape:

    stages/<stage>/base.py       abstract Strategy interface + data contracts
    stages/<stage>/<name>.py     one concrete Strategy per file
    stages/<stage>/__init__.py   REGISTRY mapping a config key -> concrete class

The orchestrator (core/recipe.py) only ever talks to the base interface and
looks concrete classes up by key in the registry. Adding a new technique to a
stage must never require editing anything outside that stage's package (plus
one registry line). See CLAUDE.md -> "Core Design Pattern".
"""
