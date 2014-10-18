# This is pretty important. Default is /bin/sh
SHELL := /bin/bash
metadata=meta.yaml

html:
	pandoc test.md --parse-raw --to html -s --mathjax -H mathjax-conf.js > test.html

pdf:
	abbreviations=$$(pandoc abbreviations.md --to latex); \
	prelims="$$(pandoc meta.yaml --template latex/prelims.tex --variable=abbreviations:"$$abbreviations" --to latex)"; \
	pandoc meta.yaml test.md -o test.pdf \
		--template latex/Thesis.tex \
		--chapter \
		--variable=prelims:"$$prelims" \
		--variable=postlims:"$$(pandoc meta.yaml --template latex/postlims.tex --to latex)"

docx:
	pandoc meta.yaml test.md --template Thesis.tex --chapter -o test.docx

clean:
	rm test.html test.pdf index.html

serve:
	ln -s test.html index.html
	python -m SimpleHTTPServer &
