from utils.data_process import AudioProcessor
from utils.streamlit_utils import plot_class_distribution, resize_image, record_sound
from utils.text_process import get_amazon_link, get_prompts
from utils.streamlit_session_flags import Flag
from gpt3_based_model.chatbot import Chatbot 
from model.inference import inferencer

import streamlit as st
import time as built_in_time
from datetime import datetime, date, time, timedelta
import torchaudio
from PIL import Image
import os 




st.set_page_config(
    page_icon="./images/Cappella_logo.png",
    layout="centered"
)
blank_image = resize_image(
        image_path="images/Title Slide.png",
        width=1600,
        height=600,
        alpha=180
    )

header_image = resize_image(
    image_path="./images/LinkedIn cover.png",
    width=1600,
    height=200,
    alpha=160
)
st.image(header_image, use_column_width=True)

header_col1, header_col2 = st.columns([2, 1])

header_col1.title("Welcome to Mini-Cappella ")
header_col1.markdown(
    "&emsp;&emsp;*-- Your personal baby-caring assistant*", unsafe_allow_html=True)

### baby infomation extraction
name_col, birthday_col = header_col1.columns([1,1])
name = header_col2.text_input(":baby::name_badge: Enter baby's name",value = "Monkey King")
min_value = datetime.now() - timedelta(days = 365)
max_value = datetime.now()
birthday = header_col2.date_input(
    ":baby::spiral_calendar_pad: Select baby's birth date",
    min_value = min_value, 
    max_value = max_value
)
baby_age = (datetime.now().date() - birthday).days
#header_col1.image("./images/capella circle.png", width=175)

times = []
for hours in range(0, 24):
  for minutes in range(0, 60):
    times.append(time(hours, minutes))

 
last_feeding_t_col, curr_t_col = header_col2.columns([1,1])
last_feeding_time = last_feeding_t_col.selectbox(
    ":alarm_clock: Enter last feeding time", 
    times,
    format_func=lambda t: t.strftime("%H:%M")
)

curr_time = curr_t_col.selectbox(":mantelpiece_clock: Adjust time to your timezone", times, key="time", format_func=lambda t: t.strftime("%H:%M"))

curr_time = datetime.combine(date.today(), curr_time)
last_feeding_time = datetime.combine(date.today(), last_feeding_time)
feeding_gap = curr_time - last_feeding_time
if feeding_gap.days < 0 :
    feeding_gap += timedelta(days = 1)
gap_h, gap_min, _ = str(feeding_gap).split(":")
if gap_h == "0" :
    gap_string = f"{gap_min} minutes"
else :
    gap_string = f"{gap_h} hours {gap_min} minutes"

pre_health_conditions = header_col2.text_area(":pill: Add pre-existing health conditions", value = "no pre-existing health conditions")
##########################################################################################################################################################
option_container = header_col1.container()
upload_info_placeholder = header_col1.empty()
option = option_container.selectbox(
    "How would you like to upload audio data?",
    ("Please choose one ","From local files", "Record on device"),
    index = 0,
    on_change = lambda :  upload_info_placeholder.empty()
)
_,record_button_pos, _ = option_container.columns([1,1,1])
if option == "From local files":
    upload_info_placeholder.empty()
    audio_data = option_container.file_uploader(label = "upload", type = ["wav","mp3"], label_visibility = "hidden")
    if audio_data is not None : 
        os.system("rm .tmp_audio.wav")
        with open(".tmp_audio.wav", "wb") as f:
            f.write(audio_data.getbuffer())
        upload_info_placeholder.success("Uploading successful!", icon="✅")

if option == "Record on device" :
    
    record_button = record_button_pos.button("Start recording")
    if record_button : 
        os.system("rm .tmp_audio.wav")
        record_sound(option_container,tmp_audio_file_path=".tmp_audio.wav")
        upload_info_placeholder.success("Uploading successful!", icon="✅")

###########################################################################################################################################################

if ".tmp_audio.wav" in os.listdir():
    audio_data = ".tmp_audio.wav"
    try:
        torchaudio.info(audio_data)
    except Exception as e:
        upload_info_placeholder.error(
            "Uploading failed ! Please check your uploaded audio data source.", icon="🚨")
        st.image(header_image, use_column_width=True)
        st.stop()

    #upload_info_placeholder.success("Uploading successful!", icon="✅")
    print(audio_data)
    st.audio(audio_data)
    
    check = st.checkbox("Review and confirm the birth date and time info to UNLOCK analysis")
    _, button_pos, _ = st.columns([1, 1, 1])

    if check and button_pos.button(" 🧠 start to analyze ... 👶🏻", use_container_width=True, type="primary"):
        
        with st.spinner("Analyzing ..."):
            audio_processor = AudioProcessor()
            audio_embedding = audio_processor.process(audio_data)
            is_baby_cry = inferencer(audio_embedding, "detection")
            if is_baby_cry:
                class_distribuiton = inferencer(
                    audio_embedding, "classification")
                print(class_distribuiton)
                most_likely_reason, freq = max(list(class_distribuiton.items()), key = lambda x : x[1])
                if freq <= 0.5 :
                    class_distribuiton = {c : 1 if c == most_likely_reason else 0 for c in class_distribuiton.keys()}
                place_holder = st.empty()
            
                prompt = get_prompts(
                    class_distribution=class_distribuiton,
                    age=baby_age,
                    curr_t=curr_time,
                    last_feeding_t=gap_string,
                    pre_conditions=pre_health_conditions
                )
                #chatbot = Chatbot(api_key=st.secrets["openai_credentials"]["personal_api_key"])
                chatbot = Chatbot(api_key="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")
                answer = chatbot.query(prompt=prompt)
                st.markdown("## Advice from AI")
                st.markdown(answer)
                
                amazon_link = get_amazon_link(class_distribution=class_distribuiton,baby_age=baby_age)
                st.markdown(f"[:shopping_trolley: click here for recommended items in Amazon :shopping_bags: ]({amazon_link})")
                
                plot_class_distribution(class_distribuiton, place_holder)
                #print(answer)
                
            else:
                st.info("Good News! No baby cry detected!")
else:
    
    st.image(blank_image, use_column_width=True)

_, writing_col, _ = st.columns([1.5, 2.1, 1.5])
writing_col.markdown("***⚙️ Powered by Cappella AI & ChatGPT 🤖***")
_, col1, col2, _ = st.columns([1, 1, 1, 1])


cappella_logo = Image.open("./images/Pink (1).png").resize((110, 110))
chatgpt_logo = Image.open(
    "./images/cdnlogo.com_ChatGPT.png").resize((100, 100))

col1.image(cappella_logo)
col2.image(chatgpt_logo)
