html:
	pandoc test.md --parse-raw --to html -s --mathjax -H mathjax-conf.js > test.html

pdf:
	pandoc meta.yaml test.md -o test.pdf \
		--template Thesis.tex \
		--chapter \
		--variable=prelims:"$$(pandoc meta.yaml --template prelims.tex --to latex)" \
		--variable=postlims:"$$(pandoc meta.yaml --template postlims.tex --to latex)"

docx:
	pandoc meta.yaml test.md --template Thesis.tex --chapter -o test.docx

clean:
	rm test.html test.pdf index.html

serve:
	ln -s test.html index.html
	python -m SimpleHTTPServer &
