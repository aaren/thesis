#!/usr/bin/env python

import os

import internalreferences as ir
import pandocfilters as pf


def make_site_links(key, value, format, metadata):
    """Converts image links from file-system-relative to
    site-relative.
    """
    if key == 'Image':
        caption, (link, title) = value
        link = os.path.join('chapters/', link)
        if os.path.isfile(link):
            link = os.path.join('/thesis', link)
            return pf.Image(caption, (link, title))


if __name__ == '__main__':
    refman = ir.ReferenceManager()
    ir.toJSONFilter([make_site_links] + refman.reference_filter)
    # ir.toJSONFilter(refman.reference_filter)
    # ir.toJSONFilter(make_site_links)
