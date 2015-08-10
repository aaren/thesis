SHELL := /bin/bash  # This is pretty important. Default is /bin/sh

metadata = meta.yaml

latex_build = build/latex
html_build = build/html
rendered = build/rendered

output_pdf = thesis.pdf

.PHONY: render all html pdf tex clean docx serve pages

all: html pdf

source := $(wildcard chapters/*.md)
source += $(wildcard chapters/*.ipynb)
# later we might want to use this:
# source_md = $(shell find chapters -name '*.md' | sort)
# but this might mean reconfiguring the output destination from render
# and the symlinking into the jekyll build

render: ${source_md} ${source_nb}
	@echo $(source)
	@mkdir -p ${rendered}
	@echo "Copying images..."
	@rsync -a chapters/figures build/rendered/
	@echo "Rendering notebooks..."
ifneq ("${source}", "")
	@$(foreach f, ${source}, notedown ${f} --render --output ${rendered}/$(basename $(notdir ${f})).md;)
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
	@echo "Building ${output_pdf}..."
	@mkdir -p build/figures
	@abbreviations=$$(pandoc abbreviations.md --to latex); \
	prelims="$$(pandoc $(metadata) \
				--template ${latex_build}/prelims.tex \
				--variable=abbreviations:"$$abbreviations" \
				--to latex)"; \
	postlims="$$(pandoc $(metadata) \
	             --template ${latex_build}/postlims.tex \
				 --to latex)"; \
	pandoc $(metadata) ${rendered}/*.md -o ${output_pdf} \
		--template ${latex_build}/Thesis.tex \
		--chapter \
		--variable=prelims:"$$prelims" \
		--variable=postlims:"$$postlims" \
        --filter ${latex_build}/filters.py

chapter: render
	pandoc $(metadata) ${rendered}/00-demo.md -o chapter.pdf \
		--template ${latex_build}/Thesis.tex \
		--chapter \
		--variable=prelims:"$$prelims" \
		--variable=postlims:"$$postlims" \
        --filter ${latex_build}/filters.py

tex: render
	@abbreviations=$$(pandoc abbreviations.md --to latex); \
	prelims="$$(pandoc $(metadata) \
				--template ${latex_build}/prelims.tex \
				--variable=abbreviations:"$$abbreviations" \
				--to latex)"; \
	postlims="$$(pandoc $(metadata) --template ${latex_build}/postlims.tex --to latex)"; \
	pandoc $(metadata) ${rendered}/*.md -o thesis.tex \
		--template ${latex_build}/Thesis.tex \
		--chapter \
		--variable=prelims:"$$prelims" \
		--variable=postlims:"$$postlims" \
        --filter ${latex_build}/filters.py

docx:
	pandoc $(metadata) test.md -o test.docx

clean:
	rm -rf ${rendered}/*

serve:
	cd ${html_build} && jekyll serve --detach --watch && cd -
