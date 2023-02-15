import torch

from model.baby_cry_classification_model.cnn_classifier import CNNClassifier
from model.baby_cry_detection_model.cnn_autoencoder import CNNAutoEncoder
from model.config import detection_model_path, classification_model_path, detection_threshold, category2numerical
numerical2category = {v: k for k, v in category2numerical.items()}


def inferencer(x, model_type):

    if model_type == "detection":
        model = CNNAutoEncoder()
        model.load_state_dict(torch.load(detection_model_path))
        model = model.cpu()

        rc_loss = model.compute_reconstruction_loss(x)
        if rc_loss <= detection_threshold:
            return True
        return False

    elif model_type == "classification":
        model = CNNClassifier()
        model.load_state_dict(torch.load(classification_model_path))
        model = model.cpu()
        logits = model(x)
        class_samples = logits.argmax(dim=-1).tolist()
        n_samples = len(class_samples)
        n_classes = len(numerical2category)
        class_counts = [0] * n_classes
        for c in class_samples:
            class_counts[c] += 1
        class_distribution = {
            numerical2category[c]: class_counts[c] / n_samples for c in range(n_classes)}
        return class_distribution

    else:
        raise TypeError(f"{model_type} is not supported")
