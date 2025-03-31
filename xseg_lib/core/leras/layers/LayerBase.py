from xseg_lib.core.leras import nn

tf = nn.tf


class LayerBase(nn.Saveable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_tensor = None

    #override
    def build_weights(self):
        pass

    #override
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        self.output_tensor = self.forward(*args, **kwargs)
        return self.output_tensor

    def get_output_tensor(self):
        """Get the output tensor from the last forward pass"""
        return self.output_tensor


nn.LayerBase = LayerBase