"""
Toy example showing a non-linear relation of the input data.
"""

from __future__ import division

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__date__ = "2016-04-30"


import inspect
import matplotlib.pyplot as plt
import sys

from os.path import abspath, dirname, join

# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))


TARGET_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', '..', 'tex', 'figures')

FIGURE_SIZE = 2.5
WIDTH = 1.0     
          
BLUE = '#3399FF'
RED = '#ED1B25'
          

fig = plt.figure()
ax = fig.add_subplot(111)
 

ax.plot([0, 6], [6, 0], color = 'k', linewidth = WIDTH)
        
ax.plot([], [])

x1 = [0.2, 4.0, 1.1, 2.4, 1.5, 3.0, 0.5, 1.8, 4.5, 1.2, 0.8, 2.2]
y1 = [1.0, 1.2, 3.9, 0.3, 1.5, 2.1, 4.2, 2.9, 0.3, 0.4, 2.2, 1.1]

x2 = [3.2, 3.3, 2.2, 0.9, 2.5, 0.3, 2.0, 3.8, 5.2, 1.4, 0.8, 2.8]
y2 = [0.5, 1.5, 1.8, 3.1, 2.7, 2.8, 3.3, 0.3, 0.2, 2.2, 1.3, 1.5]

x4 = [4.5, 1.5, 5.5, 4.0, 2.6, 6.2, 3.2, 0.5, 5.0, 6.0, 5.2, 3.5]
y4 = [2.2, 5.3, 1.2, 3.1, 4.3, 0.4, 3.8, 6.1, 1.9, 2.0, 2.8, 4.3]

x5 = [2.2, 1.2, 4.3, 2.9, 6.6, 7.2, 4.6, 1.6, 2.0, 0.3, 1.0, 3.5]
y5 = [5.6, 6.9, 3.9, 5.0, 1.3, 0.3, 3.0, 6.1, 4.8, 7.0, 5.8, 3.2]


plt.grid(axis = 'y', color = 'k')
plt.grid(axis = 'x', color = 'k')


ax.plot(x1 + x2, y1 + y2, 'ko', color = BLUE, fillstyle = 'full',
        markersize = 4, markeredgecolor = BLUE, markeredgewidth = WIDTH)
        
ax.plot(x4 + x5, y4 + y5, 'ko', color = RED, fillstyle = 'full',
        markersize = 4, markeredgecolor = RED, markeredgewidth = WIDTH)


ax.set_xlim([0, 8])         
ax.set_ylim([0, 8])
 
xmin, xmax = ax.get_xlim() 
ymin, ymax = ax.get_ylim()
 
# removing the default axis on all sides:
for side in ['bottom', 'right', 'top', 'left']:
    ax.spines[side].set_visible(False)
 
# remove the axis ticks
plt.xticks([])
plt.yticks([])
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
 
fig.set_size_inches(FIGURE_SIZE, FIGURE_SIZE)
 
# get width and height of axes object to compute matching arrowhead length and
# width
dps = fig.dpi_scale_trans.inverted()
bbox = ax.get_window_extent().transformed(dps)
width, height = bbox.width, bbox.height
 
# manual arrowhead width and length
head_width = 1/40 * (ymax - ymin) 
head_length = 1/30 * (xmax - xmin)
lw = WIDTH # axis line width
overhang = 0.7 # arrow overhang
 
# compute matching arrowhead length and width
head_width = head_width / (ymax - ymin) * (xmax - xmin) * height / width 
head_length = head_length / (xmax - xmin) * (ymax - ymin) * width / height

ax.set_aspect(1)
 
# draw x and y axis
ax.arrow(xmin, ymin, xmax - xmin, 0, fc = 'k', ec = 'k', lw = lw, 
         head_width = head_width, head_length = head_length,
         overhang = overhang, length_includes_head = True, clip_on = False) 
 
ax.arrow(xmin, ymin, 0, ymax-ymin, fc = 'k', ec = 'k', lw = lw, 
         head_width = head_width, head_length = head_length,
         overhang = overhang, length_includes_head = True, clip_on = False)

 
plt.tight_layout(0.5)

output_file_name = 'line'
plt.savefig(output_file_name + '.pdf')
#plt.savefig(join(TARGET_PATH, output_file_name + '.pgf'))

