import os
import pickle
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from xseg_lib.core.leras import nn


class XSegNet(object):
    VERSION = 1

    def __init__(self,
                 name,
                 resolution=256,
                 load_weights=True,
                 weights_file_root=None,
                 place_model_on_cpu=False,
                 run_on_cpu=False,
                 data_format="NHWC",
                 raise_on_no_model_files=False,
                 export_weights=True
                 ):

        self.resolution = resolution
        self.weights_file_root = Path(
            weights_file_root) if weights_file_root is not None else Path(
                __file__).parent

        nn.initialize(data_format=data_format)
        tf = nn.tf

        model_name = f'{name}_{resolution}'
        self.model_filename_list = []

        with tf.device('/CPU:0'):
            #Place holders on CPU
            self.input_t = tf.placeholder(
                nn.floatx, nn.get4Dshape(resolution, resolution, 3), name="input")
            self.target_t = tf.placeholder(
                nn.floatx, nn.get4Dshape(resolution, resolution, 1), name="output")

        # Initializing model classes
        with tf.device(
                '/CPU:0' if place_model_on_cpu else nn.tf_default_device_name):
            self.model = nn.XSeg(3, 32, 1, name=name)
            self.model_weights = self.model.get_weights()

        self.model_filename_list += [[self.model, f'{model_name}.npy']]

        with tf.device('/CPU:0' if run_on_cpu else nn.tf_default_device_name):
            _, pred = self.model(self.input_t)

        # Store intermediate layer tensors
        self.layer_tensors = []
        def collect_layer_tensors(layer):
            if isinstance(layer, nn.Conv2D) or isinstance(layer, nn.Conv2DTranspose):
                self.layer_tensors.append(layer.get_output_tensor())
        self.model.traverse_layers(collect_layer_tensors)
        
        # Add final output to layer tensors
        self.layer_tensors.append(pred)
        
        def net_run(input_np):
            return nn.tf_sess.run([pred], feed_dict={self.input_t: input_np})[0]
            
        def get_all_layer_outputs(input_np):
            return nn.tf_sess.run(self.layer_tensors, feed_dict={self.input_t: input_np})
            
        self.net_run = net_run
        self.get_all_layer_outputs = get_all_layer_outputs
        
        self.initialized = True
        # Loading/initializing all models weights
        for model, filename in self.model_filename_list:
            do_init = not load_weights

            if not do_init:
                model_file_path = self.weights_file_root / filename
                do_init = not model.load_weights(model_file_path)
                if do_init:
                    if raise_on_no_model_files:
                        raise Exception(f'{model_file_path} does not exists.')
                    self.initialized = False
                    break

            if do_init:
                model.init_weights()

        if export_weights:
            nn.tf.enable_resource_variables()
            nn.tf.saved_model.simple_save(nn.tf_sess, str(Path("./saved_model")), inputs={"input": self.input_t}, outputs={"output": pred})

    def get_resolution(self):
        return self.resolution

    def extract(self, input_image):
        if not self.initialized:
            return 0.5 * np.ones((self.resolution, self.resolution, 1),
                                 nn.floatx.as_numpy_dtype)

        input_shape_len = len(input_image.shape)
        if input_shape_len == 3:
            input_image = input_image[None, ...]

        result = np.clip(self.net_run(input_image), 0, 1.0)
        result[result < 0.1] = 0  #get rid of noise

        if input_shape_len == 3:
            result = result[0]

        return result

    def get_layer_outputs(self, input_image):
        """Get outputs from all intermediate layers"""
        if not self.initialized:
            return [0.5 * np.ones((self.resolution, self.resolution, 1),
                                 nn.floatx.as_numpy_dtype)]

        input_shape_len = len(input_image.shape)
        if input_shape_len == 3:
            input_image = input_image[None, ...]

        # Store intermediate layer tensors
        self.layer_tensors = []
        def collect_layer_tensors(layer):
            if isinstance(layer, nn.Conv2D) or isinstance(layer, nn.Conv2DTranspose):
                self.layer_tensors.append(layer.get_output_tensor())
        self.model.traverse_layers(collect_layer_tensors)
        
        # Add final output to layer tensors
        self.layer_tensors.append(self.model(self.input_t)[1])
        
        # Run model and get all layer outputs
        outputs = nn.tf_sess.run(self.layer_tensors, feed_dict={self.input_t: input_image})
        
        # Convert to single image if input was single image
        if input_shape_len == 3:
            outputs = [out[0] for out in outputs]
            
        return outputs
        if input_shape_len == 3:
            input_image = input_image[None, ...]

        outputs = self.get_all_layer_outputs(input_image)
        
        # Convert to single image if input was single image
        if input_shape_len == 3:
            outputs = [out[0] for out in outputs]
            
        return outputs