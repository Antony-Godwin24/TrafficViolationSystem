from ultralytics import YOLO
import torch

model = YOLO('model.pt')

def predict_violation(image_path, threshold=0.3):
    results = model(image_path)
    probs_tensor = results[0].probs.data

    if isinstance(probs_tensor, torch.Tensor):
        probs = probs_tensor.cpu().numpy().tolist()
    else:
        probs = probs_tensor

    class_names = results[0].names

    detected = [
        f"{class_names[i]} ({prob:.2f})"
        for i, prob in enumerate(probs)
        if prob > threshold
    ]

    return detected if detected else ["No violations detected"]
