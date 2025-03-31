import os
import numpy as np
import tensorflow.compat.v1 as tf
import torch
from xseg_lib.facelib.XSegNet_torch import XSegNet as TorchXSegNet

def convert_tf_to_torch(saved_model_path, output_path, resolution=256):
    # Enable TF1.x behavior
    tf.disable_v2_behavior()
    
    # Create TF session and load model
    with tf.Session() as sess:
        meta_graph = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_path)
        graph = tf.get_default_graph()
        
        # Create PyTorch model
        torch_model = TorchXSegNet()
        torch_model.eval()
        
        # Get all variables from the graph
        tf_weights = {}
        print("\nAvailable variables in saved model:")
        for var in tf.global_variables():
            print(f"Variable name: {var.name}, Shape: {var.shape}")
            tf_weights[var.name] = sess.run(var)

        # Convert weights to PyTorch format
        torch_state_dict = {}

        # Helper function to find TF variable
        def get_tf_var(name):
            if name + ':0' in tf_weights:
                return tf_weights[name + ':0']
            print(f"Warning: No match found for {name}")
            return None

        # Convert encoder blocks
        for i in range(6):  # 0-5 blocks
            # Blocks 5,4,3 have 3 convs, blocks 2,1,0 have 2 convs
            max_j = 3 if i >= 3 else 2
            for j in range(max_j):  # 0 to max_j-1
                conv_name = f'conv{i}{j+1}'
                tf_prefix = f'XSeg/{conv_name}'
                
                # Convert convolution weights
                tf_weight = get_tf_var(f'{tf_prefix}/conv/weight')
                if tf_weight is not None:
                    # Convert HWIO to OIHW
                    torch_weight = np.transpose(tf_weight, (3, 2, 0, 1))
                    torch_state_dict[f'{conv_name}.conv.weight'] = torch.from_numpy(torch_weight)

                # Convert convolution bias
                tf_bias = get_tf_var(f'{tf_prefix}/conv/bias')
                if tf_bias is not None:
                    torch_state_dict[f'{conv_name}.conv.bias'] = torch.from_numpy(tf_bias)

                # Convert FRN parameters
                tf_gamma = get_tf_var(f'{tf_prefix}/frn/weight')
                tf_beta = get_tf_var(f'{tf_prefix}/frn/bias')
                tf_eps = get_tf_var(f'{tf_prefix}/frn/eps')
                tf_tau = get_tf_var(f'{tf_prefix}/tlu/tau')
                if tf_gamma is not None:
                    torch_state_dict[f'{conv_name}.frn.gamma'] = torch.from_numpy(tf_gamma.reshape(1, -1, 1, 1))
                if tf_beta is not None:
                    torch_state_dict[f'{conv_name}.frn.beta'] = torch.from_numpy(tf_beta.reshape(1, -1, 1, 1))
                if tf_eps is not None:
                    torch_state_dict[f'{conv_name}.frn.eps'] = torch.from_numpy(tf_eps)
                if tf_tau is not None:
                    torch_state_dict[f'{conv_name}.tlu.tau'] = torch.from_numpy(tf_tau.reshape(1, -1, 1, 1))

        # Convert BlurPool filters
        pool_sizes = {1: 4, 2: 3, 3: 2, 4: 2, 5: 2, 6: 2}  # Match TF implementation
        for i, filt_size in pool_sizes.items():
            # Create BlurPool filter
            if filt_size == 2:
                a = np.array([1., 1.])
            elif filt_size == 3:
                a = np.array([1., 2., 1.])
            elif filt_size == 4:    
                a = np.array([1., 3., 3., 1.])
            
            # Create 2D filter
            a = a[:, None] * a[None, :]
            a = a[:, :, None, None]
            a = a / np.sum(a)
            
            # Register filter for each pool layer
            torch_state_dict[f'pool{i}.filter'] = torch.from_numpy(a).float()

        # Convert dense layers
        tf_dense1_weight = get_tf_var('XSeg/dense1/weight')
        tf_dense1_bias = get_tf_var('XSeg/dense1/bias')
        tf_dense2_weight = get_tf_var('XSeg/dense2/weight')
        tf_dense2_bias = get_tf_var('XSeg/dense2/bias')
        
        if tf_dense1_weight is not None:
            # TF dense weight shape: [in_features, out_features]
            # PyTorch dense weight shape: [out_features, in_features]
            torch_state_dict['dense1.weight'] = torch.from_numpy(tf_dense1_weight.T)
        if tf_dense1_bias is not None:
            torch_state_dict['dense1.bias'] = torch.from_numpy(tf_dense1_bias)
        if tf_dense2_weight is not None:
            torch_state_dict['dense2.weight'] = torch.from_numpy(tf_dense2_weight.T)
        if tf_dense2_bias is not None:
            torch_state_dict['dense2.bias'] = torch.from_numpy(tf_dense2_bias)

        # Convert decoder blocks
        for i in range(5, -1, -1):  # 5-0 blocks
            # Convert upsampling (transposed conv)
            tf_prefix = f'XSeg/up{i}'
            tf_weight = get_tf_var(f'{tf_prefix}/conv/weight')
            if tf_weight is not None:
                # Convert HWIO to OIHW for transposed conv and swap input/output channels
                torch_weight = np.transpose(tf_weight, (3, 2, 0, 1))  # HWIO -> OIHW with channels swapped
                torch_state_dict[f'up{i}.conv.weight'] = torch.from_numpy(torch_weight)
                torch_state_dict[f'up{i}.conv.bias'] = torch.from_numpy(get_tf_var(f'{tf_prefix}/conv/bias'))
                
                # Convert FRN parameters for upsampling blocks
                tf_gamma = get_tf_var(f'{tf_prefix}/frn/weight')
                tf_beta = get_tf_var(f'{tf_prefix}/frn/bias')
                tf_eps = get_tf_var(f'{tf_prefix}/frn/eps')
                tf_tau = get_tf_var(f'{tf_prefix}/tlu/tau')
                if tf_gamma is not None:
                    torch_state_dict[f'up{i}.frn.gamma'] = torch.from_numpy(tf_gamma.reshape(1, -1, 1, 1))
                if tf_beta is not None:
                    torch_state_dict[f'up{i}.frn.beta'] = torch.from_numpy(tf_beta.reshape(1, -1, 1, 1))
                if tf_eps is not None:
                    torch_state_dict[f'up{i}.frn.eps'] = torch.from_numpy(tf_eps)
                if tf_tau is not None:
                    torch_state_dict[f'up{i}.tlu.tau'] = torch.from_numpy(tf_tau.reshape(1, -1, 1, 1))

            # Convert regular convolutions in decoder block
            # Blocks 5,4,3 have 3 convs (uconv[i]3,2,1), blocks 2,1,0 have 2 convs (uconv[i]2,1)
            max_j = 3 if i >= 3 else 2
            for j in range(max_j, 0, -1):  # uconv3/2,2,1
                tf_prefix = f'XSeg/uconv{i}{j}'
                tf_weight = get_tf_var(f'{tf_prefix}/conv/weight')
                if tf_weight is not None:
                    torch_weight = np.transpose(tf_weight, (3, 2, 0, 1))
                    # TF and PyTorch uconv numbering is consistent
                    torch_state_dict[f'uconv{i}{j}.conv.weight'] = torch.from_numpy(torch_weight)
                    torch_state_dict[f'uconv{i}{j}.conv.bias'] = torch.from_numpy(get_tf_var(f'{tf_prefix}/conv/bias'))

                    # Convert FRN parameters
                    tf_gamma = get_tf_var(f'{tf_prefix}/frn/weight')
                    tf_beta = get_tf_var(f'{tf_prefix}/frn/bias')
                    tf_eps = get_tf_var(f'{tf_prefix}/frn/eps')
                    tf_tau = get_tf_var(f'{tf_prefix}/tlu/tau')
                    if tf_gamma is not None:
                        torch_state_dict[f'uconv{i}{j}.frn.gamma'] = torch.from_numpy(tf_gamma.reshape(1, -1, 1, 1))
                    if tf_beta is not None:
                        torch_state_dict[f'uconv{i}{j}.frn.beta'] = torch.from_numpy(tf_beta.reshape(1, -1, 1, 1))
                    if tf_eps is not None:
                        torch_state_dict[f'uconv{i}{j}.frn.eps'] = torch.from_numpy(tf_eps)
                    if tf_tau is not None:
                        torch_state_dict[f'uconv{i}{j}.tlu.tau'] = torch.from_numpy(tf_tau.reshape(1, -1, 1, 1))

        # Convert final output layer
        tf_weight = get_tf_var('XSeg/out_conv/weight')
        if tf_weight is not None:
            torch_weight = np.transpose(tf_weight, (3, 2, 0, 1))
            torch_state_dict['final.weight'] = torch.from_numpy(torch_weight)

        tf_bias = get_tf_var('XSeg/out_conv/bias')
        if tf_bias is not None:
            torch_state_dict['final.bias'] = torch.from_numpy(tf_bias)

        # Load converted weights into PyTorch model
        missing_keys, unexpected_keys = torch_model.load_state_dict(torch_state_dict, strict=False)
        print("\nMissing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
        
        # Save PyTorch model
        torch.save(torch_model.state_dict(), output_path)
        print(f"\nModel saved to {output_path}")
        return torch_model

if __name__ == "__main__":
    # Example usage
    saved_model_path = "../saved_model"  # Directory containing saved_model.pb
    output_path = "../xseg_torch.pth"
    model = convert_tf_to_torch(saved_model_path, output_path)
