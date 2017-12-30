
# Kyle Hart
# Project: Commodities vs Weather
# Date: Dec 10, 2017
#
# Description: A visualization for the results produced and saved by the NaiveBayes models applied to commodities data.
#              Horizontal bar graph showing the accuracy of the models for the given commodities.
#              Bar width is indicative of number of countries producing a given commodity.
#              Color bar on side correlates with number of transactions ran by the model for each good.

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as clr
import matplotlib.cm as cm
from pyspark import SparkContext
from pyspark.sql import SparkSession
import os
import pandas as pd

''' Credit where credit is due. I relied heavily on this particular blog post
    https://datasciencelab.wordpress.com/2013/12/21/beautiful-plots-with-pandas-and-matplotlib/
    
    I adjusted accordingly for my own data, and never quite got the colors as pretty as they did'''

# Get Spark context
os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
sc = SparkContext("local[4]", "ratings filter")
sc.setLogLevel(logLevel="OFF")
spark = SparkSession(sparkContext=sc)

results_df = spark.read.csv('data/results_df.csv', header=True, inferSchema=True)
results_df.orderBy('accuracy', ascending=False).show(n=100, truncate=False)

# Shape data into pandas dataframe for plotting
items = []
accuracies = []
transactions = []
countries = []
t_min = results_df.select('transactions').rdd.min()[0]
t_max = results_df.select('transactions').rdd.max()[0]
for row in results_df.toLocalIterator():
    items.append(row['item'])
    accuracies.append(row['accuracy'])
    transactions.append(row['transactions']/t_max)
    countries.append(row['countries'])

data = { 'accuracies' : pd.Series(accuracies, index=items),
         'transactions' : pd.Series(transactions, index=items),
         'countries': pd.Series(countries, index=items) }

df = pd.DataFrame(data)


# Plot figure set title
fig = plt.figure(figsize=(12, 24))
ax = fig.add_subplot(111)
title = 'Crops and Livestock as Predictors of Income Level'

# Set transparency, make colormap
trans = 1
custom_cmap = cm.YlGnBu(df['transactions'])

# Plot horizontal bars of accuracies
df['accuracies'].plot( kind='barh', ax=ax, alpha=trans, legend=False,
                      color=custom_cmap, edgecolor='w', xlim=(0,.8), title=title)
# Make pretty
ax.grid(False)
ax.set_frame_on(False)


# Set title
ax.set_title(ax.get_title(), fontsize=22, alpha=trans, ha='left')
plt.subplots_adjust(top=1)
ax.title.set_position((-0.2, 1.025))

# Set axis labels
ax.xaxis.set_label_position('top')
x_label = 'Accuracy'
ax.set_xlabel(x_label, fontsize=12, ha='left')
ax.xaxis.set_label_coords(-0.08, 1.005)

# Adjust ticks on axis
ax.xaxis.tick_top()
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks_position('none')

xticks = [.1, .2, .3, .4, .5]
ax.xaxis.set_ticks(xticks)
ax.set_xticklabels(xticks, fontsize=8, alpha=trans)


# Set item labels
yticks = [item.get_text() for item in ax.get_yticklabels()]
ax.set_yticklabels(yticks, fontsize=8, alpha=trans)
ax.yaxis.set_tick_params(pad=20 )

# Set min and max bar thickness (from 0 to 1)
hmin, hmax = 0.3, 0.9
xmin, xmax = min(df['countries']), max(df['countries'])
# Function that interpolates linearly between hmin and hmax
f = lambda x: hmin + (hmax - hmin) * (x - xmin) / (xmax - xmin)
# Make array of heights
hs = [f(x) for x in df['countries']]

# Iterate over bars
for container in ax.containers:
    # Each bar has a Rectangle element as child
    for i, child in enumerate(container.get_children()):
        # Reset the lower left point of each bar so that bar is centered
        child.set_y(child.get_y() - 0.125 + 0.5 - hs[i] / 2)
        # Attribute height to each Recatangle according to number of countries producing item
        plt.setp(child, height=hs[i])

# Legend
# Create fake labels for legend
line1 = Line2D([], [], linewidth=9, color='k', alpha=trans)
line2 = Line2D([], [], linewidth=18, color='k', alpha=trans)

# Set two legend labels to be min and max of counties of production
labels = [min(df['countries']), max(df['countries'])]

# Position legend in lower right part
# Set ncol=2 for horizontally expanding legend
legend = ax.legend([line1, line2], labels, ncol=2, frameon=False, fontsize=16,
                bbox_to_anchor=[.7, 0.11], handlelength=2,
                handletextpad=1, columnspacing=2, title='Production in # Countries')

# Customize legend title
# Set position to increase space between legend and labels
plt.setp(legend.get_title(), fontsize=10, alpha=trans)
legend.get_title().set_position((0, 10))
# Customize transparency for legend labels
[plt.setp(label, alpha=trans) for label in legend.get_texts()]

# Create a fake colorbar
ctb = clr.LinearSegmentedColormap.from_list('custombar', custom_cmap)
# Trick from http://stackoverflow.com/questions/8342549/
# matplotlib-add-colorbar-to-a-sequence-of-line-plots
sm = plt.cm.ScalarMappable(cmap=ctb, norm=clr.Normalize(vmin=min(df['transactions']), vmax=max(df['transactions'])))
# Fake up the array of the scalar mappable
sm._A = []

# Set colorbar, aspect ratio
cbar = plt.colorbar(sm, alpha=0.5, aspect=16, shrink=0.4)
cbar.solids.set_edgecolor("face")
# Remove colorbar container frame
cbar.outline.set_visible(False)
# Fontsize for colorbar ticklabels
cbar.ax.tick_params(labelsize=8)
# Customize colorbar tick labels
mytks = range(int(t_min/50), int(t_max/50))
cbar.set_ticks(mytks)
cbar.ax.set_yticklabels([str(a) for a in mytks], alpha=trans)

# Colorbar label, customize fontsize and distance to colorbar
cbar.set_label('Model Transactions', alpha=trans,
               rotation=270, fontsize=12, labelpad=12)

# Remove color bar tick lines, while keeping the tick labels
cbarytks = plt.getp(cbar.ax.axes, 'yticklines')
plt.setp(cbarytks, visible=False)

# Save figure in png with tight bounding box
plt.savefig('visualization.png', bbox_inches='tight', dpi=300)
