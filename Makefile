PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: report test lint

report:
	$(PY) -m tca_groundtruth.report --out reports

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
