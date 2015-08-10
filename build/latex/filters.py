import os
import re
import base64

import pandocfilters as pf
import internalreferences as ir


def suppress_input_cells(key, value, format, metadata):
    """Suppresses code cells that have the attribute '.input'.
    """
    if format == 'latex' and key == 'CodeBlock':
        return pf.Null()


def raw_equations(key, value, format, metadata):
    """Pass latex contained in $$...$$ to latex writer
    as raw latex *if* the latex starts with 'begin'.
    """
    if format == 'latex' and key == 'Math' and value[0]['t'] == 'DisplayMath':
        mathtype, content = value
        if content.strip().startswith('\\begin{'):
            # n.b, has to be rawinline as the element we are replacing
            # is inline
            return pf.RawInline('latex', content)


def isOutputFigure(key, value):
    return key == 'Div' and 'figure' in value[0][1] and 'output' in value[0][1]


png_uri = re.compile(r'data:image/png;base64,([0-9a-zA-Z/+=]*)')
image_count = 0


def save_uri(key, value, format, metadata):
    if format == 'latex' and isOutputFigure(key, value):
        attr, blocks = value
        image = blocks[0]['c'][0]  # should be only inline in only block in the div
        caption, (uri, title) = image['c']

        global image_count
        image_count += 1
        fname = 'build/figures/{i:05d}.png'.format(i=image_count)

        match = png_uri.search(uri)
        if match:
            data = match.groups()[0]
        else:
            return

        with open(fname, 'wb') as f:
            f.write(base64.b64decode(data))

        return pf.Div(attr, [pf.Plain([pf.Image(caption, (fname, title))])])


def make_site_links(key, value, format, metadata):
    """Converts image links from file-system-relative to
    site-relative.
    """
    if key == 'Image':
        caption, (link, title) = value
        link = os.path.join('chapters/', link)
        if os.path.isfile(link):
            return pf.Image(caption, (link, title))


if __name__ == '__main__':
    refman = ir.ReferenceManager()
    # raw_equations has to come last as it returns raw latex
    ir.toJSONFilter([make_site_links, save_uri, suppress_input_cells]
                    + refman.reference_filter
                    + [raw_equations])
