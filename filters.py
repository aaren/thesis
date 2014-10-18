import pandocfilters as pf


def raw_equations(key, value, format, metadata):
    """Pass latex contained in $$...$$ to latex writer
    as raw latex.
    """
    if key == 'Math' and 1:
        mathtype, content = value
        return pf.RawInline('latex', content)

if __name__ == '__main__':
    pf.toJSONFilter(raw_equations)
