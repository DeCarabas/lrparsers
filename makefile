# The first test just makes sure we don't have any particular dependencies at
# all, so it can be dropped into a random unrelated project for build
# purposes.
.PHONY: test
test:
	python3 ./parser/parser.py
	uv run python3 -m pytest

.PHONY: dep
dep: lrparser.mk

lrparser.mk: makedep.py pyproject.toml
	python3 makedep.py

include lrparser.mk

.PHONY: wheel
wheel: dist/lrparsers-$(VERSION)-py3-none-any.whl

dist/lrparsers-$(VERSION).tar.gz dist/lrparsers-$(VERSION)-py3-none-any.whl: pyproject.toml $(PYTHON_SOURCES)
	uv build --no-clean

.PHONY: clean
clean:
	rm -rf ./dist

DINGUS_FILES=\
	dingus/srvit.py \
	dingus/index.html \
	dingus/dingus.js \
	dingus/worker.js \
	dingus/style.css \
	dingus/codemirror/codemirror.css \
	dingus/codemirror/codemirror.js \
	dingus/codemirror/python.js \
	dingus/pyodide/micropip-0.6.0-py3-none-any.whl \
	dingus/pyodide/micropip-0.6.0-py3-none-any.whl.metadata \
	dingus/pyodide/packaging-23.2-py3-none-any.whl \
	dingus/pyodide/packaging-23.2-py3-none-any.whl.metadata \
	dingus/pyodide/pyodide.asm.js \
	dingus/pyodide/pyodide.asm.wasm \
	dingus/pyodide/pyodide-core-0.26.2.tar \
	dingus/pyodide/pyodide.d.ts \
	dingus/pyodide/pyodide.js \
	dingus/pyodide/pyodide-lock.json \
	dingus/pyodide/pyodide.mjs \
	dingus/pyodide/python_stdlib.zip \

DINGUS_TARGETS=$(addprefix dist/, $(DINGUS_FILES))

.PHONY: dingus
dingus: $(DINGUS_TARGETS) dist/dingus/wheel/lrparsers-$(VERSION)-py3-none-any.whl dist/dingus/about.html
	python3 ./dist/dingus/srvit.py

dist/dingus/%: dingus/%
	mkdir -p $(dir $@)
	ln $< $@

dist/dingus/about.html: dingus/about.md
	pandoc $< -o $@ -s

dist/dingus/wheel/lrparsers-$(VERSION)-py3-none-any.whl: dist/lrparsers-$(VERSION)-py3-none-any.whl
	mkdir -p $(dir $@)
	cp $< $@
