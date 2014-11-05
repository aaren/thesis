SHELL := /bin/bash  # This is pretty important. Default is /bin/sh

metadata=meta.yaml

latex_build=build/latex
html_build=build/html
rendered=build/rendered

.PHONY: render all html pdf tex clean docx serve pages

all: html pdf

source_md := $(wildcard chapters/*.md)
source_nb := $(wildcard chapters/*.ipynb)

render: $(source_md) $(source_nb)
	@mkdir -p ${rendered}
ifneq ("$(source_md)", "")
	notedown $(source_md) --render --output ${rendered}/$(basename $(notdir $(source_md))).md
endif
ifneq ("$(source_nb)", "")
	notedown $(source_nb) --render --output ${rendered}/$(basename $(notdir $(source_nb))).md
endif

html: render
	@echo "Building html..."
	@cd ${html_build} && jekyll build && cd -

# build _site and push diff to gh-pages branch
# using a temporary git repo to rebase the changes onto
pages: html
	@echo "Building gh-pages..."
	@root_dir=$$(git rev-parse --show-toplevel) && \
	tmp_dir=$$(mktemp -d) && \
	cd $${tmp_dir} && git init --quiet && \
	git remote add origin $${root_dir} && \
	git pull --quiet origin gh-pages && \
	git rm -rf --cached --quiet * && \
	rsync -a $${root_dir}/${html_build}/_site/ . && \
	git add -A && git commit --quiet -m "update gh-pages" && \
	git push --quiet origin master:gh-pages

# remember each line in the recipe is executed in a *new* shell,
# so if we want to pass variables around we have to make a long
# single line command.
pdf: render
	@echo "Building pdf..."
	@abbreviations=$$(pandoc abbreviations.md --to latex); \
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
	rm ${rendered}/*

serve:
	cd ${html_build} && jekyll serve --detach --watch && cd -
