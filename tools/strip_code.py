#!/home/eeaol/src/anaconda/bin/python

import pandocfilters as pf


def strip_code(key, value, format, meta):
    if key in ('CodeBlock', 'Header'):
        return pf.Null()
    elif key in ('DisplayMath'):
        return pf.Space()


if __name__ == '__main__':
    pf.toJSONFilter(strip_code)
