import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
import os
from functools import wraps
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
import logging


logger = logging.getLogger(__name__)

def add_colorbar(fig, ax, label='sentiment', cmap='RdYlBu', vmin=-1, vmax=1, x=0, y=0, length=.2, width=.01, labelsize=None, norm=None, fmt=None):
    if norm is None:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cax = fig.add_axes([x, y, length, width])
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, cax=cax, orientation='horizontal', shrink=1)
    cbar.ax.tick_params(axis='x', direction='out', labelsize=5)
    if fmt is not None:
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter(fmt))
    cbar.set_label(label)
    cbar.outline.set_visible(False)

def find_project_root():
    return os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', '..'))

def notebook_plot(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        style_sheet = os.path.join(find_project_root(), 'analysis', 'stylesheets', 'notebook.mplstyle')
        plt.style.use(style_sheet)
        return func(*args, **kwargs)
    return wrapper

def paper_plot(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        style_sheet = os.path.join(find_project_root(), 'analysis', 'stylesheets', 'paper.mplstyle')
        plt.style.use(style_sheet)
        return func(*args, **kwargs)
    return wrapper

def use_stylesheet(name='paper'):
    style_sheet = os.path.join(find_project_root(), 'analysis', 'stylesheets', '{}.mplstyle'.format(name))
    plt.style.use(style_sheet)

def label_subplots(axes, upper_case=True, offset_points=(-40, 0)):
    start_ord = 65 if upper_case else 97
    for ax, lab in zip(axes, ['{}'.format(chr(j)) for j in range(start_ord, start_ord + len(axes))]):
        ax.annotate(lab, (0, 1), xytext=offset_points, xycoords='axes fraction', textcoords='offset points',
                ha='right', va='top', weight='bold')

def save_fig(fig, name, plot_formats=['png', 'eps'], version=1, dpi=300):
    def f_name(fmt):
        f_name = '{}.{}'.format(name, fmt)
        return os.path.join(folder_path, f_name)
    folder_path = os.path.join(find_project_root(), 'analysis', 'plots', name, 'v{}'.format(version))
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    for fmt in plot_formats:
        if not fmt == 'tiff':
            f_path = f_name(fmt)
            logger.info('Writing figure file {}'.format(os.path.abspath(f_path)))
            fig.savefig(f_name(fmt), bbox_inches='tight', dpi=dpi)
    if 'tiff' in plot_formats:
        os.system("convert {} {}".format(f_name('png'), f_name('tiff')))

def format_time_axis(ax, ticks='weekly', major_ticks_out=True, minor_ticks_out=True):
    if ticks == 'daily':
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif ticks == 'weekly':
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    elif ticks == 'monthly':
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    if minor_ticks_out:
        ax.tick_params(axis='x', direction='out', which='minor', zorder=2, size=2)
    if major_ticks_out:
        ax.tick_params(axis='x', direction='out', which='major', zorder=2, size=4)
    return ax
