This is the source for my PhD thesis.

You write your chapters in either markdown or IPython notebooks - it
doesn't matter which.

The build script can produce html, latex and pdf output.

To build everything:

    make

To build pdf:

    make pdf

To build html

    make html

The templates for html and latex output are contained in `build/`.

The latex template is made to satisfy the University of Leeds
formatting requirements.

The html template is a themed Jekyll website. It can be served
locally with

    make serve

and will then be visible at 'localhost:4000/thesis'.

There is a `gh-pages` branch that is automatically populated with
the Jekyll build following a commit to master.

User specific configuration is done in `meta.yaml` and in
`build/html/_config.yml'.
