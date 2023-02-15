detection_model_path = "./model/saved_model/CNNAE_12layers_with_noise_valid.pt"
classification_model_path = "./model/saved_model/CNNclf_4layers_final.pt"
detection_threshold = 40
category2numerical = {
    "hunger" : 0,
    "pain" : 1,
    "burping" : 2,
    "discomfort" : 3,
    "tired" : 4
}