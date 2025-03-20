#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 07:18:13 2022

@author: Ryan McClarren
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 08:26:20 2022

@author: bennett
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import interpolate
from scipy import integrate
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt
from math import pi
import matplotlib.font_manager as fm
import matplotlib.ticker as mtick
from matplotlib.ticker import ScalarFormatter
from scipy import optimize
# from labellines import labelLine, labelLines
import csv
from matplotlib.ticker import StrMethodFormatter, NullFormatter

# font = fm.FontProperties(family = 'Gill Sans', fname = '/users/wbenn/Anaconda3/Library/Fonts/GillSans.ttc', size = 20)
# axisfont = fm.FontProperties(family = 'Gill Sans', fname = '/users/wbenn/Anaconda3/Library/Fonts/GillSans.ttc', size = 14)
axisfont = fm.FontProperties(size = 20);
font = fm.FontProperties(size = 18);


matplotlib.rcParams['pdf.fonttype'] = 42;
matplotlib.rcParams['ps.fonttype'] = 42;
def hide_spines(choose_ticks, ticks,intx=False,inty=False):
    """Hides the top and rightmost axis spines from view for all active
    figures and their respective axes."""

    # Retrieve a list of all current figures.
    figures = [x for x in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()];
    if (plt.gca().get_legend()):
        plt.setp(plt.gca().get_legend().get_texts()); 
    for figure in figures:
        # Get all Axis instances related to the figure.
        for ax in figure.canvas.figure.get_axes():
            # Disable spines.
            ax.spines['right'].set_color('none');
            ax.spines['top'].set_color('none');
            # Disable ticks.
            ax.xaxis.set_ticks_position('bottom');
            ax.yaxis.set_ticks_position('left');
            
            
            # ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10)) ))
            # ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("v" % v)))
            # ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'));
            if choose_ticks == True:
                ax.set_xticks(ticks);
            
            
            
            for label in ax.get_xticklabels() :
                label.set_fontproperties(font);
            for label in ax.get_yticklabels() :
                label.set_fontproperties(font);
            # ax.set_xticklabels(ax.get_xticks(), fontproperties = font)
            ax.set_xlabel(ax.get_xlabel() );
            ax.set_ylabel(ax.get_ylabel());
            ax.set_title(ax.get_title());
            if (inty):
                ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'));
            if (intx):
                ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'));
def show_loglog(nm,xlimleft,xlimright,a=0,b=0, choose_ticks = False, ticks = [0,1,2]):
    hide_spines(choose_ticks, ticks, a,b);
    # plt.locator_params(axis = 'x', nbins=4)
    # plt.locator_params(axis = 'y', nbins=4)
    plt.xlim(xlimleft,xlimright);
    plt.minorticks_off();
    # ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10)) ))
    # plt.yticks([1,1e-2,1e-4,1e-6,1e-8,1e-10,1e-12], labels)
    #ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10)) ))
    if (len(nm)>0):
        plt.savefig(nm+".pdf",bbox_inches='tight');
    plt.show();
    

    
    
    
    
# x_data = [2,4,8,16,32]

# y_data = [0.1, 0.1/2, 0.1/4, 0.1/8, 0.1/16]


# plt.loglog(x_data, y_data)

# show_loglog("test.pdf")






