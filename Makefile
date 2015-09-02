SHELL := /bin/bash  # This is pretty important. Default is /bin/sh

metadata = meta.yaml

latex_build = build/latex
html_build = build/html
rendered = build/rendered
which_rendered = *.md

# the base name of the output file from pandoc (e.g. thesis.pdf, thesis.tex)
output = thesis

.PHONY: render all html pdf tex clean docx serve pages

all: html pdf

source := $(wildcard chapters/0*.md)
source += $(wildcard chapters/0*.ipynb)
# later we might want to use this:
# source_md = $(shell find chapters -name '*.md' | sort)
# but this might mean reconfiguring the output destination from render
# and the symlinking into the jekyll build

abbreviations = abbreviations="$$(pandoc abbreviations.md --to latex)"
packages = packages="$$(pandoc $(metadata) --template $(latex_build)/packages.tex --to latex)"
title = title="$$(pandoc $(metadata) --template $(latex_build)/title.tex --to latex)"
prelims = prelims="$$(pandoc $(metadata) --template $(latex_build)/prelims.tex --variable=abbreviations:"$$abbreviations" --to latex)"
postlims = postlims="$$(pandoc $(metadata) --template $(latex_build)/postlims.tex --to latex)"

# single chapter creation mode (set chapter=chapters/blah.md on command line)
ifneq ("$(chapter)", "")
	source = $(chapter)
	fname = "$(basename $(notdir $(chapter)))"
	output = $(fname)
	which_rendered = "$(fname).md"

	title = title=""
	prelims = prelims='\tableofcontents'
	postlims = postlims=""
endif

# remember each line in the recipe is executed in a *new* shell,
# so if we want to pass variables around we have to make a long
# single line command. This is using gmake 3.81, which doesn't
# have .ONESHELL.
define pandoc
	$(packages); \
	$(title); \
	$(abbreviations); \
	$(prelims); \
	$(postlims); \
	pandoc $(metadata) $(rendered)/$(which_rendered) -o $(1) \
		--template $(latex_build)/Thesis.tex \
		--chapter \
		--variable=packages:"$$packages" \
		--variable=title:"$$title" \
		--variable=prelims:"$$prelims" \
		--variable=postlims:"$$postlims" \
        --filter $(latex_build)/filters.py
endef


render: $(source)
	@echo $(source)
	@mkdir -p $(rendered)
	@echo "Copying images..."
	@rsync -a chapters/figures build/rendered/
	@echo "Rendering notebooks..."
ifneq ("$(source)", "")
	@$(foreach f, $(source), notedown $(f) --render --output $(rendered)/$(basename $(notdir $(f))).md;)
endif

html: render
	@echo "Building html..."
	@cd $(html_build) && jekyll build && cd -

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
	rsync -a $${root_dir}/$(html_build)/_site/ . && \
	git add -A && git commit --quiet -m "update gh-pages" && \
	git push --quiet origin master:gh-pages

pdf: render
	@echo "Building $(output).pdf..."
	@mkdir -p build/figures
	@$(call pandoc,"$(output).pdf")

tex: render
	@echo "Building $(output).tex..."
	@$(call pandoc,"$(output).tex")

docx: render
	@echo "Building $(output).docx..."
	@$(call pandoc,"$(output).docx")

clean:
	rm -rf $(rendered)/* ./*pdf ./*tex ./*docx

serve:
	cd $(html_build) && jekyll serve --detach --watch && cd -
