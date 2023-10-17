from pytorch_quantization import tensor_quant
from absl import logging as quant_logging
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules
import mmcv.cnn.bricks.wrappers
from pytorch_quantization import nn as quant_nn

import torch
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from typing import Callable

# 1.4 拷贝initialize，缺的包导入
def initialize():
    quant_logging.set_verbosity(quant_logging.ERROR)
    quant_desc_input = QuantDescriptor(calib_method="histogram")

    quant_modules._DEFAULT_QUANT_MAP.append(
        quant_modules._quant_entry(mmcv.cnn.bricks.wrappers, "ConvTranspose2d", quant_nn.QuantConvTranspose2d)
    )

    for item in quant_modules._DEFAULT_QUANT_MAP:
        item.replace_mod.set_default_quant_desc_input(quant_desc_input)

    quant_logging.set_verbosity(quant_logging.ERROR)
  
  
def transfer_torch_to_quantization(nninstance : torch.nn.Module, quantmodule):

    quant_instance = quantmodule.__new__(quantmodule)
    for k, val in vars(nninstance).items():
        setattr(quant_instance, k, val)

    def __init__(self):
        quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            quant_desc_input = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True)
            self.init_quantizer(quant_desc_input)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            self.init_quantizer(quant_desc_input, quant_desc_weight)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instance)
    return quant_instance  
    
def replace_to_quantization_module(model : torch.nn.Module):

    module_dict = {}
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)
        module_dict[id(module)] = entry.replace_mod

    def recursive_and_replace_module(module, prefix=""):
        for name in module._modules:
            submodule = module._modules[name]
            path      = name if prefix == "" else prefix + "." + name
            recursive_and_replace_module(submodule, path)

            submodule_id = id(type(submodule))
            if submodule_id in module_dict:  
                module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])

    recursive_and_replace_module(model)
    
    
def set_quantizer_fast(module): 
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
             if isinstance(module._calibrator, calib.HistogramCalibrator):
                module._calibrator._torch_hist = True 
                
                
                
                
                
from tqdm import tqdm            
def calibrate_model(model : torch.nn.Module, images, device, batch_processor_callback: Callable = None, num_batch=1):

    def compute_amax(model, **kwargs):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax(strict=False)
                    else:
                        module.load_calib_amax(strict=False, **kwargs)

                    module._amax = module._amax.to(device)
        
    def collect_stats(model, images, device, num_batch=2):
        """Feed data to the network and collect statistics"""
        # Enable calibrators
        model.eval()
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        iter_count = 0 
        for data in tqdm(images, total=num_batch, desc="Collect stats for calibrating"):
            with torch.no_grad():
                result = model(data)
            iter_count += 1
            if iter_count >num_batch:
                break

        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

    collect_stats(model, images, device, num_batch=num_batch)
    compute_amax(model, method="mse")                