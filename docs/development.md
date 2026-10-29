# Development

[English](development.md) | [简体中文](development.zh-CN.md)

The core requires Python 3.10+ and the standard library. In your own checkout and Python environment:

```text
python -m pip install -e .
python -X utf8 -m unittest discover -s tests -t . -v
```

Tests use temporary databases, deterministic test doubles and fake learners. They check provenance, bounded state, transaction rollback, version retirement, snapshot isolation and recovery. Test fixtures live under `tests/fixtures`, are not installed with the package, and are not experiment data or benchmark results. No model download, API credentials or external service is required.

The optional `runtime` extra supplies psutil for resource monitoring. Optional native environment wrappers require separately managed dependencies/assets. There is no experiment CLI or selected model backend.

When adding an application backend, validate evidence admission, material exclusions, token accounting, cancellation and candidate loading before publishing a version. In particular, test that failed training leaves the selected external memory intact and that old tasks keep their pinned version. Real hardware limits and scientific evaluation require separate acceptance.

Source installation and tests validate engineering contracts only; they do not establish model quality, training effectiveness or benchmark performance.
