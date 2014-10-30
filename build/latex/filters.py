import pandocfilters as pf
import internalreferences as ir


def suppress_input_cells(key, value, format, metadata):
    """Suppresses code cells that have the attribute '.input'.
    """
    if format == 'latex' and key == 'CodeBlock' and 'input' in value[0][1]:
        return pf.Null()


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
    ir.toJSONFilter(refman.reference_filter +
                    [raw_equations, suppress_input_cells])
