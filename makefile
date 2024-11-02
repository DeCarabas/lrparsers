# The first test just makes sure we don't have any particular dependencies at
# all, so it can be dropped into a random unrelated project for build
# purposes.
.PHONY: test
test:
	python3 ./parser/parser.py
	pdm run python3 -m pytest

.PHONY: dep
dep: lrparser.mk

lrparser.mk: makedep.py pyproject.toml
	python3 makedep.py

include lrparser.mk

.PHONY: wheel
wheel: dist/lrparsers-$(VERSION)-py3-none-any.whl

dist/lrparsers-$(VERSION).tar.gz dist/lrparsers-$(VERSION)-py3-none-any.whl: pyproject.toml $(PYTHON_SOURCES)
	pdm build

.PHONY: clean
clean:
	rm -rf ./dist
	rm -rf ./dingus/wheel/*
	rm ./dingus/about.html

# TODO: Get the built dingus artifacts out of the tree :P
#       Use hard-links to make editing pleasant.

.PHONY: dingus
dingus: dingus/wheel/lrparsers-$(VERSION)-py3-none-any.whl dingus/about.html
	python3 ./dingus/srvit.py

dingus/about.html: dingus/about.md
	pandoc $< -o $@ -s

dingus/wheel/lrparsers-$(VERSION)-py3-none-any.whl: dist/lrparsers-$(VERSION)-py3-none-any.whl
	cp $< $@
