clean:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	rm -rf .mypy_cache .pytest_cache build dist *.egg-info
