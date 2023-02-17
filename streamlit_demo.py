from utils.data_process import AudioProcessor
from model.inference import inferencer
import streamlit as st
import time
import numpy as np
from matplotlib import pyplot as plt
import torchaudio
from PIL import Image

plt.rcParams['axes.edgecolor'] = '#333F4B'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333F4B'
plt.rcParams['ytick.color'] = '#333F4B'


st.set_page_config(
    page_icon="./images/Cappella_logo.png",
    layout="centered"
)

image = Image.open("./images/LinkedIn cover.png")
new_image = image.resize((1600, 200))
new_image.putalpha(160)
st.image(new_image, use_column_width=True)

header_col1, header_col2 = st.columns([2, 1])

header_col1.title("Welcome to Cappella ")
header_col1.markdown(
    "&emsp;*-- Your personal baby-caring assistant*", unsafe_allow_html=True)
header_col2.markdown(" ")
header_col2.markdown(" ")
header_col2.image("./images/capella circle.png", width=175)

col1, col2 = st.columns([2, 1])

audio_data = col1.file_uploader(label="upload here", type=[
                                "wav", "mp3"], label_visibility='hidden')


if audio_data:
    try:
        torchaudio.load(audio_data)
    except:
        st.error("Uploading failed ! Please check your uploaded audio data source.", icon="🚨")
        st.image("images/Title Slide.png")
        st.stop()
    col2.markdown(" ")
    col2.markdown(" ")
    col2.markdown(" ")
    col2.success("Uploading successful!",icon="✅")

    st.audio(audio_data)
    _,button_pos,_ = st.columns([1,1,1])
   
    if button_pos.button(" 🧠 start to analyze ... 👶🏻", use_container_width = True, type = "primary"):
        with st.spinner("Analyzing ..."):
            time.sleep(1)
            audio_processor = AudioProcessor()
            audio_embedding = audio_processor.process(audio_data)

            is_baby_cry = inferencer(audio_embedding, "detection")
            if is_baby_cry:
                class_distribuiton = inferencer(
                    audio_embedding, "classification")
                print(class_distribuiton)

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

                col1, col2 = st.columns([2, 1])
                col1.pyplot(fig)
            else:
                st.info("Good News! No baby cry detected!")
else:
    
    image = Image.open("images/Title Slide.png") 
    new_image = image.resize((1600, 600))
    new_image.putalpha(180)
    st.image(new_image, use_column_width=True)

_,writing_col,_ = st.columns([1.5,2.1,1.5])
writing_col.markdown("***⚙️ Powered by Cappella AI & ChatGPT 🤖***")
_,col1, col2,_ = st.columns([1,1,1,1])

cappella_logo = Image.open("./images/Pink (1).png").resize((110,110))
chatgpt_logo = Image.open("./images/cdnlogo.com_ChatGPT.png").resize((100,100))

col1.image(cappella_logo)
col2.image(chatgpt_logo)