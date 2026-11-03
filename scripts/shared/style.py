"""What the plots and tables of the thesis share."""

TEXT_WIDTH_POINTS = 345
TEXT_WIDTH_INCHES = TEXT_WIDTH_POINTS / 72.27
BLUE, ORANGE = '#226aa6', '#c36a20'

PLOT_STYLE = {
    'font.family': 'serif', 'font.serif': ['STIX Two Text', 'STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
    'lines.linewidth': 1.2, 'pdf.fonttype': 42,
}

# In the order of the tables.
MESH_NAMES = {
    'stanford-bunny-low': 'Bunny (low)', 'stanford-bunny': 'Bunny', 'armadillo-r128': 'Armadillo',
    'stanford-dragon-r128': 'Dragon', 'fandisk': 'Fandisk', 'max-planck': 'Max Planck', 'horse': 'Horse',
    'rocker-arm': 'Rocker arm', 'homer': 'Homer', 'elk': 'Elk',
}

# The symbols of Sec. 4.1 as table headers; the captions say what they measure.
LOSS_SYMBOL, TIME_SYMBOL = r'$\bar\lambda$', '$t$'
MIN_ANGLE_SYMBOL, ANGLE_TAIL_SYMBOL = r'$\theta_{\min}$', r'$\theta_{5\%}$'
VALENCE_SYMBOL, DEVIATION_SYMBOL = r'$\sigma_{\mathrm{val}}$', r'$d_{\mathrm{RMS}}$'

# The colours of the schematic figures, as LaTeX defines them.
PALETTE = {
    'opInk': '303C44', 'opNeutral': 'F1F4F5', 'opBefore': 'D77D43', 'opBeforeFill': 'FAEADD', 'opOperand': '9E3C12',
    'opAfter': '35A39E', 'opAfterFill': 'DCEFED', 'opFail': 'C0392B', 'opPass': '1E8449', 'schedInk': '303C44',
    'schedQueue': 'F1F4F5',
}


def decimal_column(cells):
    """A siunitx column that aligns the cells at the decimal point and fits all of them; a cell may start with <."""
    comparator = '<' if any(cell.startswith('<') for cell in cells) else ''
    parts = [cell.lstrip('<').partition('.') for cell in cells]
    integer, decimals = max(len(whole) for whole, _, _ in parts), max(len(fraction) for _, _, fraction in parts)
    return f'S[table-format={comparator}{integer}.{decimals}]'


def define_color(name):
    return rf'\definecolor{{{name}}}{{HTML}}{{{PALETTE[name]}}}'
