import os
import cv2
import numpy as np
import torch
import tensorflow as tf
from xseg_lib.facelib.XSegNet_torch import XSegNet
from xseg_lib import get_xseg

def get_tf_model_outputs(img, tf_model):
    """Get intermediate outputs from TensorFlow model"""
    xseg_res = tf_model.get_resolution()
    
    # Preprocess image (following apply_xseg logic)
    img = img.astype(np.float32) / 255.0
    if img.shape[1] != xseg_res:
        img = cv2.resize(img, (xseg_res, xseg_res), interpolation=cv2.INTER_LANCZOS4)
    
    if len(img.shape) == 2:
        img = img[..., None]
    
    # Get intermediate outputs
    outputs = tf_model.get_layer_outputs(img)
    
    # Get final output and apply thresholding
    final_output = outputs[-1].copy()
    final_output[final_output < 0.5] = 0
    final_output[final_output >= 0.5] = 1
    
    # Save intermediate outputs
    for i, out in enumerate(outputs):
        np.save(f'tf_layer_{i}_output.npy', out)
    
    return outputs

def get_torch_model_outputs(img, model_path, resolution):
    """Get intermediate outputs from PyTorch model"""
    # Load model
    model = XSegNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Store intermediate outputs
    layer_outputs = []
    def hook_fn(module, input, output):
        # Convert output to numpy array
        if isinstance(output, torch.Tensor):
            out_np = output.detach().cpu().numpy()
        elif isinstance(output, tuple):
            out_np = output[0].detach().cpu().numpy()
        layer_outputs.append(out_np)
    
    # Register hooks for all conv and convtranspose layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Preprocess image
    img = img.astype(np.float32) / 255.0
    if img.shape[0] != resolution or img.shape[1] != resolution:
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
    img = img.transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img).unsqueeze(0)
    
    # Forward pass
    with torch.no_grad():
        output = model(img_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Convert final output to numpy and apply thresholding
    output_np = output.cpu().numpy()
    output_np = output_np.copy()
    output_np[output_np < 0.5] = 0
    output_np[output_np >= 0.5] = 1
    layer_outputs.append(output_np)
    
    # Save intermediate outputs
    for i, out in enumerate(layer_outputs):
        np.save(f'torch_layer_{i}_output.npy', out)
    
    return layer_outputs

def compare_outputs(tf_outputs, torch_outputs):
    """Compare outputs from both models layer by layer"""
    print("\nComparing layer outputs:")
    print("-" * 50)
    
    # Compare each layer's output
    min_layers = min(len(tf_outputs), len(torch_outputs))
    
    # Create a directory for layer visualizations
    os.makedirs('layer_comparisons', exist_ok=True)
    
    def save_feature_maps(output, prefix):
        # For multi-channel outputs, save each channel separately
        if len(output.shape) == 3 and output.shape[-1] > 1:
            for c in range(output.shape[-1]):
                channel = output[..., c]
                # Normalize to 0-255 range
                channel_norm = ((channel - channel.min()) / (channel.max() - channel.min() + 1e-6) * 255).astype(np.uint8)
                cv2.imwrite(f'layer_comparisons/{prefix}_channel_{c}.png', channel_norm)
        else:
            # For single channel, just save directly
            if len(output.shape) == 3:
                output = output[..., 0]  # Take first channel if multi-channel
            output_norm = ((output - output.min()) / (output.max() - output.min() + 1e-6) * 255).astype(np.uint8)
            cv2.imwrite(f'layer_comparisons/{prefix}.png', output_norm)
    
    for i in range(min_layers):
        tf_out = tf_outputs[i]
        torch_out = torch_outputs[i]
        
        # Convert PyTorch output (NCHW) to TF format (NHWC)
        if torch_out.ndim == 4:  # NCHW format
            torch_out = np.transpose(torch_out[0], (1, 2, 0))  # Convert to HWC
        
        print(f"\nLayer {i}:")
        print(f"TF output shape: {tf_out.shape}")
        print(f"PyTorch output shape: {torch_out.shape}")
        
        try:
            # Try to match shapes if needed
            if tf_out.shape != torch_out.shape:
                print(f"Warning: Shape mismatch in layer {i}")
                # Save the feature maps anyway for inspection
                save_feature_maps(tf_out, f'layer_{i}_tf')
                save_feature_maps(torch_out, f'layer_{i}_torch')
                continue
            
            # Calculate differences
            abs_diff = np.abs(tf_out - torch_out)
            max_diff = np.max(abs_diff)
            mean_diff = np.mean(abs_diff)
            std_diff = np.std(abs_diff)
            
            # Calculate channel-wise statistics if multi-channel
            if len(tf_out.shape) == 3 and tf_out.shape[-1] > 1:
                print(f"\nChannel-wise statistics:")
                for c in range(tf_out.shape[-1]):
                    ch_diff = np.abs(tf_out[..., c] - torch_out[..., c])
                    print(f"Channel {c}:")
                    print(f"  Max diff: {np.max(ch_diff):.6f}")
                    print(f"  Mean diff: {np.mean(ch_diff):.6f}")
                    print(f"  Std diff: {np.std(ch_diff):.6f}")
            
            print(f"\nOverall Statistics:")
            print(f"Max difference: {max_diff:.6f}")
            print(f"Mean difference: {mean_diff:.6f}")
            print(f"Std difference: {std_diff:.6f}")
            
            # Save visualizations
            save_feature_maps(tf_out, f'layer_{i}_tf')
            save_feature_maps(torch_out, f'layer_{i}_torch')
            save_feature_maps(abs_diff, f'layer_{i}_diff')
            
            if max_diff > 0.1:
                print(f"WARNING: Large differences detected in layer {i}")
                print(f"Check layer_comparisons/layer_{i}_* for visualizations")
        
        except Exception as e:
            print(f"Error comparing layer {i}: {str(e)}")
            continue
    
    print(f"\nTotal layers compared: {min_layers}")
    if len(tf_outputs) != len(torch_outputs):
        print(f"Warning: Number of layers differ - TF: {len(tf_outputs)}, PyTorch: {len(torch_outputs)}")
    print(f"\nLayer visualizations saved in 'layer_comparisons' directory")

if __name__ == "__main__":
    # Load and preprocess image
    img_path = "../weights/tgt.jpg"
    model_path = "../xseg_torch.pth"
    img = cv2.imread(img_path)
    
    print("Loading TensorFlow model...")
    tf_model = get_xseg()
    resolution = tf_model.get_resolution()
    
    print("Running TensorFlow model...")
    tf_outputs = get_tf_model_outputs(img, tf_model)
    
    print("\nRunning PyTorch model...")
    torch_outputs = get_torch_model_outputs(img, model_path, resolution)
    
    # Compare outputs
    compare_outputs(tf_outputs, torch_outputs)
