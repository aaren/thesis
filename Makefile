SHELL := /bin/bash  # This is pretty important. Default is /bin/sh

metadata=meta.yaml

latex_build=./build/latex
html_build=./build/html
rendered=./build/rendered

.PHONY: render all html pdf tex clean docx serve

all: html pdf

source_md := $(wildcard chapters/*.md)
source_nb := $(wildcard chapters/*.ipynb)

render: $(source_md)
	notedown --render $(source_md) --output ${rendered}/$(basename $(notdir $(source_md))).md

html: render
	cd ${html_build} && jekyll build && cd -

# remember each line in the recipe is executed in a *new* shell,
# so if we want to pass variables around we have to make a long
# single line command.
pdf: render
	abbreviations=$$(pandoc abbreviations.md --to latex); \
	prelims="$$(pandoc $(metadata) \
				--template ${latex_build}/prelims.tex \
				--variable=abbreviations:"$$abbreviations" \
				--to latex)"; \
	postlims="$$(pandoc $(metadata) --template ${latex_build}/postlims.tex --to latex)"; \
	pandoc $(metadata) ${rendered}/* -o thesis.pdf \
		--template ${latex_build}/Thesis.tex \
		--chapter \
		--variable=prelims:"$$prelims" \
		--variable=postlims:"$$postlims" \
        --filter ${latex_build}/filters.py

tex: render
	abbreviations=$$(pandoc abbreviations.md --to latex); \
	prelims="$$(pandoc $(metadata) \
				--template ${latex_build}/prelims.tex \
				--variable=abbreviations:"$$abbreviations" \
				--to latex)"; \
	postlims="$$(pandoc $(metadata) --template ${latex_build}/postlims.tex --to latex)"; \
	pandoc $(metadata) ${rendered}/* -o thesis.tex \
		--template ${latex_build}/Thesis.tex \
		--chapter \
		--variable=prelims:"$$prelims" \
		--variable=postlims:"$$postlims" \
        --filter ${latex_build}/filters.py

docx:
	pandoc $(metadata) test.md -o test.docx

clean:
	rm test.pdf test.tex ${rendered}/*

serve:
	cd ${html_build} && jekyll serve --detach --watch && cd -
