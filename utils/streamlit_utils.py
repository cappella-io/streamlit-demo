import numpy as np    
import matplotlib.pyplot as plt      
import streamlit as st
from PIL import Image
import pyaudio 
import wave

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

def record_sound(
    position : st,
    length : int = 5,
    n_channels : int = 1,
    buffer_size : int = 1024,
    sample_rate : int = 22050,
    tmp_audio_file_path : str = ".tmp_audio.wav",
    
    
):
    
    progress_bar = position.progress(0)
    
    FORMAT = pyaudio.paInt16
    N_CHANNEL = n_channels
    CHUNKSIZE = buffer_size 
    MAX_SECONDS = length
    SAMPLE_RATE = sample_rate

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=N_CHANNEL,
        rate = SAMPLE_RATE,
        input = True,
        frames_per_buffer=CHUNKSIZE,
    )
    
    n_chunk_per_second = round(SAMPLE_RATE / CHUNKSIZE)
    frame_bytes = []
    frame_arrays = []
    for i in range(MAX_SECONDS) :
        for j in range(n_chunk_per_second) : 
            frame_byte = stream.read(CHUNKSIZE)
            frame_array = list(np.fromstring(frame_byte))
            frame_bytes.append(frame_byte)
            frame_arrays.extend(frame_array)
        
        progress_bar.progress(int((i + 1) / MAX_SECONDS * 100))
    
    stream.stop_stream()
    stream.close()
    p.terminate()

    sound_file = wave.open(tmp_audio_file_path, "wb")
    sound_file.setnchannels(1)
    sound_file.setsampwidth(p.get_sample_size(FORMAT))
    sound_file.setframerate(SAMPLE_RATE)
    sound_file.writeframes(b"".join(frame_bytes))
    sound_file.close()
    
    
    