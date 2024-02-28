import sys
import os
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


input_path = sys.argv[1]
output_path = sys.argv[2]
config_path = sys.argv[3]

# 判断权重是否是需要 control 的
# 简单的说，在模型中以 control_ 开头的权重都是需要 control 的，如 control_model.middle_block_out.0.bias
# 在模型中指的是 self.control_model
# 如果是的话，返回 True, "model.middle_block_out.0.bias"
def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

# 加载模型
configs = OmegaConf.load(config_path)
model = instantiate_from_config(configs["model"])
scratch_dict = model.state_dict()

# 加载预训练权重
pretrained_weights = torch.load(input_path)['state_dict']

# 生成目标权重
target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_')
    # copy_k 指的是原始的权重
    if is_control:
        copy_k = 'model.diffusion_' + name 
        # 需要 control 的
        # model.diffusion_model.middle_block_out.0.bias
        # 对应的是 self.model.diffusion_model
    else:
        copy_k = k      
        # 不需要 control 的 
    
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
        # pbe 中存在的权重，从中拷贝
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')
        # pbe 中不存在的权重，置空


# 保存新的权重
model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')