## Checks

Run before every commit; CI runs the same on each pull request:

```bash
ruff check .
ruff format --check .
pytest
```

Read `CONTEXT.md` for domain vocabulary and `docs/adr/` for past decisions.

## Agent skills

### Issue tracker

Issues and specs are tracked with GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the five canonical triage labels. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository. See `docs/agents/domain.md`.
