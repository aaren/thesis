---
layout: post
title: Automatically building github pages with Jekyll
---

I'm working with a file structure like this on master:

```
thesis/
    chapters/
    papers/
    build/
        latex/
        html/  <-- Jekyll project for this site
```

I want this site to be published on github pages, but I can't let Github build
it for me because I'm using a custom Jekyll config (using pandoc).

I also don't want to track the `_site` folder in the Jekyll project on my
master branch, and I'd like for my site to be automatically built each time
I commit to master.

Starting from an answer on StackOverflow I wrote this post-commit hook to do
this (stored at `.git/hooks/post-commit`):


```bash
#!/bin/bash

# derived from
# https://stackoverflow.com/questions/23097489/git-map-directory-to-branch/23097670#23097670

# assumes that you have a jekyll project located in $HTML on master and that
# you want to build and publish only the '_site' of this project to a gh-pages
# branch.

# Uses a temporary build branch to remove the need to track _site on master.

HTML="build/html"    # the location of the Jekyll project in master
site="$HTML/_site/"  # the location of _site

source=master   # source branch (default master)
dest=gh-pages   # destination branch (default gh-pages)
build=_tmp_build     # temporary build branch

current_dir="$(pwd)"
root_dir="$(git rev-parse --show-toplevel)"

cd ${root_dir}

# create temporary build branch
if _=`git rev-parse -q --verify ${build}`; then
    echo "build branch exists, aborting."
    exit
fi

git checkout -b ${build}

cd ${HTML}
jekyll build
cd ${root_dir}
git add -f ${site}
git commit -m "built site"

if ghpages=`git rev-parse -q --verify ${dest}`; then
        # there's already a branch, don't double-commit a tree
        committed=`git rev-parse -q --verify ${dest}^{tree}`
        current=`git rev-parse -q --verify ${build}:${site}`
        test x$current = x$committed && exit
fi

if commit=`git commit-tree ${ghpages:+-p $ghpages} -m "updating $dest" ${build}:${site}`; then
        git update-ref refs/heads/${dest} ${commit}
fi


# clean up
git checkout ${source}
git branch -D ${build}
cd ${current_dir}
```

Now when I commit a change to master, my site gets built and pushed to
gh-pages.
