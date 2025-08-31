import torch
import cv2
import numpy as np
import time
from pathlib import Path


def export_to_jit(model_path, jit_model_path, example_input_shape=(1, 256, 256,3), trace_mode=True):
    """
    Export PyTorch model to JIT format
    
    Args:
        model_path: Path to the PyTorch model (.pth)
        jit_model_path: Path to save the JIT model
        example_input_shape: Shape of example input for tracing
        trace_mode: If True, use torch.jit.trace, otherwise use torch.jit.script
    
    Returns:
        Path to the saved JIT model
    """
    print(f"Loading model from {model_path}...")
    model = torch.load(model_path, weights_only=False)
    model.eval()
    
    # Move model to CPU for export (can be moved to GPU later during inference)
    model = model.cpu()
    
    # Create example input for tracing
    example_input = torch.randn(example_input_shape)
    
    print(f"Exporting model to JIT format using {'trace' if trace_mode else 'script'} mode...")
    
    if trace_mode:
        # Use tracing - captures the model structure based on example inputs
        jit_model = torch.jit.trace(model, example_input)
    else:
        # Use scripting - analyzes the model code to create a serializable version
        jit_model = torch.jit.script(model)
    
    # Save the JIT model
    jit_model.save(jit_model_path)
    print(f"JIT model saved to {jit_model_path}")
    
    return jit_model_path


def test_jit_model(image_path, jit_model_path, resolution=256):
    """
    Test the exported JIT model
    
    Args:
        image_path: Path to the test image
        jit_model_path: Path to the JIT model
        resolution: Input resolution for the model
    """
    # Load JIT model
    print(f"Loading JIT model from {jit_model_path}...")
    model = torch.jit.load(jit_model_path)
    model.eval()
    
    # Check if CUDA is available and move model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = img.astype(np.float32) / 255.0
    if img.shape[0] != resolution or img.shape[1] != resolution:
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
    
    # Convert to tensor and ensure correct format (NCHW)
    # Assuming the model expects NCHW format
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    
    # Inference with timing
    print("Running inference...")
    inference_times = []
    for i in range(10):
        t0 = time.time()
        with torch.no_grad():
            pred = model(img_tensor)
        t1 = time.time()
        inference_time = t1 - t0
        inference_times.append(inference_time)
        print(f"Inference {i+1}/10: {inference_time:.4f} seconds")
    
    avg_time = sum(inference_times) / len(inference_times)
    print(f"Average inference time: {avg_time:.4f} seconds")
    
    # Post-process
    pred = pred.squeeze().cpu().numpy()
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1
    pred = (pred * 255).astype(np.uint8)
    
    # Save result
    result_path = 'result_jit.png'
    cv2.imwrite(result_path, pred)
    print(f"Result saved to {result_path}")


if __name__ == "__main__":
    # Paths
    model_path = "./xseg_torch.pth"
    jit_model_path = "./xseg_torch_jit.pt"
    image_path = "weights/tgt.jpg"
    
    # Export model to JIT format
    export_to_jit(model_path, jit_model_path)
    
    # Test the exported JIT model
    test_jit_model(image_path, jit_model_path)
