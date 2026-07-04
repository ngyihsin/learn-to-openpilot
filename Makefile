.PHONY: help install test grade status notebooks clean

help:
	@echo "learn-openpilot — make targets:"
	@echo "  make install    Install Python dependencies"
	@echo "  make status     Show your 30-day progress map"
	@echo "  make test       Run every auto-grader (all days)"
	@echo "  make grade D=1  Grade a single day (e.g. D=8)"
	@echo "  make notebooks  Regenerate the sample lesson notebooks"
	@echo "  make clean      Remove caches and generated artifacts"

install:
	pip install -r requirements.txt

test:
	pytest

grade:
	@python tools/grade.py day $(D)

status:
	@python tools/grade.py status

notebooks:
	python tools/build_lessons.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} +
