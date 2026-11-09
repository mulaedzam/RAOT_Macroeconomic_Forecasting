# Environment reproduction

The manuscript states that the exact software/package versions from the archived experiment environment should be reported before submission. Those exact versions were not available in the saved project files used to assemble this repository package, so they are intentionally **not invented here**.

On the original experiment machine/environment, run:

```bash
python tools/capture_environment.py
python -m pip freeze > environment/requirements-lock.txt
```

Then commit `environment/environment_snapshot.json` and `environment/requirements-lock.txt` before the public release/archival DOI is created.

The unpinned `requirements.txt` only records the packages imported by the released analysis scripts; it is not a substitute for the original lock file.
