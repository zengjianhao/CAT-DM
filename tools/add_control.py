import sys
import os
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

input_path = sys.argv[1]
output_path = sys.argv[2]
config_path = sys.argv[3]

'''
# Determine whether the weight needs to be controlled?
# In simple terms, in the model, all weights that begin with "control_" need to be controlled. For instance, "control_model.middle_block_out.0.bias" need to be controlled.
# In code,  "control_model.middle_block_out.0.bias" belongs to "self.control_model"
# Return True, "model.middle_block_out.0.bias"
'''

def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

# Load model
configs = OmegaConf.load(config_path)
model = instantiate_from_config(configs["model"])
scratch_dict = model.state_dict()

# Load pre-trained weights
pretrained_weights = torch.load(input_path)['state_dict']

# Generate target weights
target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_')
    # Need to be controlled
    if is_control:
        copy_k = 'model.diffusion_' + name
    # Don't need to be controlled
    else:
        copy_k = k
    
    # The weights that exist in pbe, copy from it
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    # The weights not existing in pbe, set to zero
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')


# Save
model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')