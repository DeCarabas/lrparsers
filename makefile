# The first test just makes sure we don't have any particular dependencies at
# all, so it can be dropped into a random unrelated project for build
# purposes.
.PHONY: test
test:
	python3 ./parser/parser.py
	pdm run pytest
