
import datetime
url_templetes = {
    "hunger": lambda age: f"https://www.amazon.com/s?k=baby+food+{age}+months+old&sprefix=baby+food+for+{age}+old%2Caps%2C351&ref=nb_sb_ss_ts-doa-p_1_19",
    "pain": lambda age: f"https://www.amazon.com/s?k=baby+pain+relief+for+4+months+old&crid=3F8YFDAKLM201&sprefix=baby+pain+relief+f+{age}+months+old%2Caps%2C490&ref=nb_sb_noss",
    "discomfort": lambda age: f"https://www.amazon.com/s?k=baby+Discomfort+relief+for+{age}+months+old&crid=1A9X8Z8BF14VS&sprefix=baby+discomfort+relief+for+4+months+old%2Caps%2C341&ref=nb_sb_noss",
    "diaper": lambda age: f"https://www.amazon.com/s?k=baby+diaper+for+4+months+old&crid=ZSPTVYTN2CSJ&sprefix=baby+diaper+for+{age}+months+old%2Caps%2C339&ref=nb_sb_noss_2",
    "burping": lambda age: f"https://www.amazon.com/s?k=baby+Burping+at+{age}+months+old&crid=1KGEL01RRUAMP&sprefix=baby+burping+at+{age}+months+old%2Caps%2C338&ref=nb_sb_noss",
    "tired": lambda age: f"https://www.amazon.com/s?k=baby+sleep+at+4+months+old&crid=1GLYRTYCYWDCJ&sprefix=baby+tired+at+{age}+months+old%2Caps%2C1546&ref=nb_sb_noss",
    "hugs": lambda age: f"https://www.amazon.com/s?k=baby+hugs+cozy+{age}+months+old&crid=34AD4O190KSG0&sprefix=baby+hugs+cozy+{age}+months+old%2Caps%2C438&ref=nb_sb_noss "
}


def get_amazon_link(
    class_distribution: dict,
    baby_age: int
):
    cry_reason = max(list(class_distribution.items()), key=lambda x: x[1])[0]
    return url_templetes[cry_reason](baby_age)


prompt_templetes = [
    lambda cry_reason, curr_t, last_feeding_t, age, pre_conditions:
        f"Act as if you are an expert on the first couple of months of life and you are giving new parents personalized advice. Your integrated baby cry translator technology has identified that the baby is crying because it is {cry_reason}. It is currently {curr_t} and the last feeding was {last_feeding_t} ago. The baby is {age} days old and has {pre_conditions}. Write a long paragraph giving a new parent personalized advice in that situation."
]


def get_prompts(
    class_distribution: dict,
    age: int,
    curr_t: datetime.time,
    last_feeding_t: str,
    pre_conditions: str,
    prompt_id: int = 0
):
    cry_reason = max(list(class_distribution.items()), key=lambda x: x[1])[0]
    return prompt_templetes[prompt_id](cry_reason, curr_t, last_feeding_t, age, pre_conditions)
