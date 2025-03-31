import cv2
import numpy as np
import torch

def test_torch_model(image_path, model_path, resolution=256):
    # Load model
    model = torch.load(model_path, weights_only=False)
    model.eval()
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = img.astype(np.float32) / 255.0
    if img.shape[0] != resolution or img.shape[1] != resolution:
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)

    # Convert to tensor
    img_tensor = torch.from_numpy(img).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        pred = model(img_tensor)
    
    # Post-process
    pred = pred.squeeze().numpy()
    pred[pred < 0.5] = 0  # Match TF threshold
    pred[pred >= 0.5] = 1
    pred = (pred * 255).astype(np.uint8)
    
    # Save result
    cv2.imwrite('result_torch.png', pred)

if __name__ == "__main__":
    # Example usage
    image_path = "weights/tgt.jpg"  # Replace with your test image
    model_path = "./xseg_torch.pth"
    test_torch_model(image_path, model_path)
