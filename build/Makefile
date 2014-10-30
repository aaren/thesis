# This is pretty important. Default is /bin/sh
SHELL := /bin/bash
metadata=meta.yaml

.PHONY: html pdf clean docx serve

html:
	cd html && jekyll build && cd ..

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
        --filter latex/filters.py

docx:
	pandoc $(metadata) test.md -o test.docx

clean:
	rm test.html test.pdf index.html

serve:
	python -m SimpleHTTPServer &
