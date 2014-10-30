# This is pretty important. Default is /bin/sh
SHELL := /bin/bash
metadata=meta.yaml

latex_build=./build/latex
html_build=./build/html

.PHONY: html pdf clean docx serve

html:
	cd ${html_build} && jekyll build && cd -

# remember each line in the recipe is executed in a *new* shell,
# so if we want to pass variables around we have to make a long
# single line command.
pdf:
	abbreviations=$$(pandoc abbreviations.md --to latex); \
	prelims="$$(pandoc $(metadata) \
				--template ${latex_build}/prelims.tex \
				--variable=abbreviations:"$$abbreviations" \
				--to latex)"; \
	postlims="$$(pandoc $(metadata) --template ${latex_build}/postlims.tex --to latex)"; \
	pandoc $(metadata) test.md -o test.pdf \
		--template ${latex_build}/Thesis.tex \
		--chapter \
		--variable=prelims:"$$prelims" \
		--variable=postlims:"$$postlims" \
        --filter ${latex_build}/filters.py

docx:
	pandoc $(metadata) test.md -o test.docx

clean:
	rm test.pdf

serve:
	cd ${html_build} && jekyll serve --detach --watch && cd -
