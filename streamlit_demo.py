import streamlit as st
import time 
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams['axes.edgecolor']='#333F4B'
plt.rcParams['axes.linewidth']=0.8
plt.rcParams['xtick.color']='#333F4B'
plt.rcParams['ytick.color']='#333F4B'

from model.inference import inferencer
from utils.data_process import AudioProcessor

st.set_page_config(layout="wide")
col1, col2 = st.columns([1,2])
col1.title("Welcome to Cappella ")
col1.markdown("&emsp;*-- Your personal baby-caring assistant*", unsafe_allow_html = True)
audio_data = col2.file_uploader(label = "upload here",type = ["wav","mp3"], label_visibility = 'hidden')
if audio_data :
    st.success("uploading successful!")
    st.audio(audio_data)
    if st.button("Start to analyze") :
        with st.spinner("Analyzing ...") :
            time.sleep(1)
            audio_processor = AudioProcessor()
            audio_embedding = audio_processor.process(audio_data)
            is_baby_cry = inferencer(audio_embedding, "detection")
            if is_baby_cry :
                class_distribuiton = inferencer(audio_embedding, "classification")
                print(class_distribuiton)
                
                c, freq = zip(*class_distribuiton.items())
                freq = np.array(freq) + 0.01
                y_pos = np.arange(len(c))
                fig, ax = plt.subplots(figsize = (8,4))
                ax.hlines(y = y_pos, xmin = 0, xmax = freq, color='#007acc', alpha=0.2, linewidth=10)
                ax.plot(freq, y_pos, "o", markersize=10,color='#007acc', alpha=0.6)
                ax.invert_yaxis()
                ax.set_yticks(y_pos, labels=c)
                
                ax.set_xlabel('Percentage', fontsize=15, fontweight='black', color = '#333F4B')
                ax.set_ylabel('Reasons for crying', fontsize=15, fontweight='black', color = '#333F4B')
                
                # change the style of the axis spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                #ax.spines['left'].set_visible(False)
                #ax.spines['bottom'].set_visible(False)
                #ax.spines['left'].set_bounds((-0.1, len(y_pos) - 1))
                ax.set_xlim(-0.1,1.1)
                # add some space between the axis and the plot
                ax.spines['left'].set_position(('outward', 5))
                ax.spines['bottom'].set_position(('outward', 5))
                plt.style.use("seaborn")
                
                col1, col2 = st.columns([2,1])
                col1.pyplot(fig)
            else :
                st.info("Good News! No baby cry detected!")
    

