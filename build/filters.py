import pandocfilters as pf
import internalreferences as ir


def raw_equations(key, value, format, metadata):
    """Pass latex contained in $$...$$ to latex writer
    as raw latex.
    """
    if key == 'Math' and format == 'latex':
        mathtype, content = value
        # n.b, has to be rawinline as the element we are replacing
        # is inline
        return pf.RawInline('latex', content)

if __name__ == '__main__':
    refman = ir.ReferenceManager()
    ir.toJSONFilter(refman.reference_filter + [raw_equations])
