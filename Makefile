# This is pretty important. Default is /bin/sh
SHELL := /bin/bash
metadata=meta.yaml

.PHONY: html pdf clean docx serve

html:
	pandoc test.md --parse-raw --to html -s --mathjax -H html/mathjax-conf.js > index.html

# remember each line in the recipe is executed in a *new* shell,
# so if we want to pass variables around we have to make a long
# single line command.
pdf:
	abbreviations=$$(pandoc abbreviations.md --to latex); \
	prelims="$$(pandoc $(metadata) \
				--template latex/prelims.tex \
				--variable=abbreviations:"$$abbreviations" \
				--to latex)"; \
	postlims="$$(pandoc $(metadata) --template latex/postlims.tex --to latex)"; \
	pandoc $(metadata) test.md -o test.pdf \
		--template latex/Thesis.tex \
		--chapter \
		--variable=prelims:"$$prelims" \
		--variable=postlims:"$$postlims" \
        --filter filters.py

docx:
	pandoc $(metadata) test.md --template Thesis.tex --chapter -o test.docx

clean:
	rm test.html test.pdf index.html

serve:
	python -m SimpleHTTPServer &
