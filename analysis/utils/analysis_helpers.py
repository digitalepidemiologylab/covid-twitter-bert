import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
import json
import os
from functools import wraps
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
import logging
import glob
import pandas as pd


logger = logging.getLogger(__name__)
# disable matplotlib font manage logger
logging.getLogger('matplotlib.font_manager').disabled = True

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

def get_run_logs(pattern=None, run_type='finetune', bucket_name='my-bucket', project_name='covid-bert'):
    f_names = glob.glob(os.path.join(find_project_root(), 'data', bucket_name, project_name, run_type, '*', 'run_logs.json'))
    df = []
    for f_name in f_names:
        if pattern is None or pattern in f_name.split('/')[-2]:
            with open(f_name, 'r') as f:
                df.append(json.load(f))
    df = pd.DataFrame(df)
    if len(df) > 0:
        df['created_at'] = pd.to_datetime(df.created_at)
        df.sort_values('created_at', inplace=True, ascending=True)
    return df

def get_summary_files(pattern=None, run_type='pretrain', bucket_name='my-bucket', project_name='covid-bert'):
    f_names = glob.glob(os.path.join(find_project_root(), 'data', bucket_name, project_name, run_type, '*', 'summaries', 'train', '*'))
    files = []
    for f_name in f_names:
        if pattern is None or pattern in f_name.split('/')[-4]:
            files.append(f_name)
    return files

def find_project_root():
    return os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', '..'))

def plot(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        style_sheet = os.path.join(find_project_root(), 'analysis', 'stylesheets', 'figure_small.mplstyle')
        plt.style.use(style_sheet)
        return func(*args, **kwargs)
    return wrapper

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
