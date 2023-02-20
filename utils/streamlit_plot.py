import numpy as np    
import matplotlib.pyplot as plt      
import streamlit as st
from PIL import Image
plt.rcParams['axes.edgecolor'] = '#333F4B'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333F4B'
plt.rcParams['ytick.color'] = '#333F4B'


def plot_class_distribution(class_distribuiton, position = st) :
    
    c, freq = zip(*class_distribuiton.items())
    freq = np.array(freq) + 0.01
    y_pos = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hlines(y=y_pos, xmin=0, xmax=freq,
                color='#007acc', alpha=0.2, linewidth=10)
    ax.plot(freq, y_pos, "o", markersize=10,
            color='#007acc', alpha=0.6)
    ax.invert_yaxis()
    ax.set_yticks(y_pos, labels=c)

    ax.set_xlabel('Percentage', fontsize=15,
                    fontweight='black', color='#333F4B')
    ax.set_ylabel('Reasons for crying', fontsize=15,
                    fontweight='black', color='#333F4B')

    # change the style of the axis spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_bounds((-0.1, len(y_pos) - 1))
    ax.set_xlim(-0.1, 1.1)
    # add some space between the axis and the plot
    ax.spines['left'].set_position(('outward', 5))
    ax.spines['bottom'].set_position(('outward', 5))
    plt.style.use("seaborn")

    position.pyplot(fig)

def resize_image(image_path, width, height, alpha = 0) :
    
    image = Image.open(image_path)
    new_image = image.resize((width, height))
    new_image.putalpha(alpha)
    
    return image 