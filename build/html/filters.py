#!/usr/bin/env python

import os

import internalreferences as ir
import pandocfilters as pf
from pandocattributes import PandocAttributes


collapse_button = """
<input type="checkbox" id="codeblock-{n}" class="toggle">
    <label for="codeblock-{n}">Source code [{n}]</label>
""".format


def make_site_links(key, value, format, metadata):
    """Converts image links from file-system-relative to
    site-relative.
    """
    if key == 'Image':
        caption, (link, title) = value
        test_link = os.path.join('_chapters/', link)
        if os.path.isfile(test_link):
            link = os.path.join('/thesis/chapters', link)
            return pf.Image(caption, (link, title))


def enclose_input_code(key, value, format, metadata):
    """Enclose any input code in a div to allow collapsing.

    'Input code' is a code block that has 'n' defined (the prompt
    number).
    """
    if key == 'CodeBlock':
        div_attr = {'classes': ['collapsible']}
        code_attr = PandocAttributes(value[0], format='pandoc')
        if 'n' not in code_attr.kvs:
            return
        else:
            button = pf.RawBlock('html', collapse_button(n=code_attr['n']))
            code = pf.CodeBlock(*value)
            content = [button, pf.Div(pf.attributes({}), [code])]
            return pf.Div(pf.attributes(div_attr), content)


if __name__ == '__main__':
    refman = ir.ReferenceManager()
    ir.toJSONFilter([make_site_links, enclose_input_code] + refman.reference_filter)
    # ir.toJSONFilter(refman.reference_filter)
    # ir.toJSONFilter(make_site_links)
