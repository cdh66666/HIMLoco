# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil
import torch.nn.functional as F

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    """
    解析模拟参数，根据命令行参数和配置字典来初始化和设置 Isaac Gym 的模拟参数。

    参数:
    args (object): 包含命令行参数的对象。
    cfg (dict): 包含模拟配置的字典。

    返回:
    gymapi.SimParams: 初始化并设置好的模拟参数对象。
    """
    # 代码来源于 Isaac Gym Preview 2
    # 初始化模拟参数对象
    sim_params = gymapi.SimParams()

    # 根据命令行参数设置一些模拟参数
    if args.physics_engine == gymapi.SIM_FLEX:
        # 如果使用 Flex 物理引擎且设备不是 CPU，给出警告
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        # 如果使用 PhysX 物理引擎，设置是否使用 GPU 以及子场景数量
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    # 设置是否使用 GPU 流水线
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # 如果配置字典中包含 "sim" 键，解析其中的模拟选项并更新或覆盖之前设置的参数
    if "sim" in cfg:
        # 调用 gymutil.parse_sim_config 函数解析配置字典中的模拟参数
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # 如果命令行中指定了线程数且使用 PhysX 物理引擎，则覆盖模拟参数中的线程数
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    # 返回设置好的模拟参数对象
    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    """
    根据命令行参数更新环境配置和训练配置。

    参数:
    env_cfg (object): 环境配置对象。
    cfg_train (object): 训练配置对象。
    args (object): 命令行参数对象。

    返回:
    tuple: 包含更新后的环境配置和训练配置的元组。
    """
    # 处理随机种子
    if env_cfg is not None:
        # 处理环境数量
        if args.num_envs is not None:
            # 如果命令行指定了环境数量，更新环境配置中的环境数量
            env_cfg.env.num_envs = args.num_envs
        if args.seed is not None:
            # 如果命令行指定了随机种子，更新环境配置中的随机种子
            env_cfg.seed = args.seed
    if cfg_train is not None:
        if args.seed is not None:
            # 如果命令行指定了随机种子，更新训练配置中的随机种子
            cfg_train.seed = args.seed
        # 处理算法运行器的参数
        if args.max_iterations is not None:
            # 如果命令行指定了最大迭代次数，更新训练配置中的最大迭代次数
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            # 如果命令行指定了恢复训练，更新训练配置中的恢复训练标志
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            # 如果命令行指定了实验名称，更新训练配置中的实验名称
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            # 如果命令行指定了运行名称，更新训练配置中的运行名称
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            # 如果命令行指定了要加载的运行名称，更新训练配置中的加载运行名称
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            # 如果命令行指定了检查点编号，更新训练配置中的检查点编号
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "go1", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    # args.sim_device_id = args.compute_device_id
    args.sim_device = args.rl_device
    # if args.sim_device=='cuda':
    #     args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'estimator'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterHIM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

# class PolicyExporterLSTM(torch.nn.Module):
#     def __init__(self, actor_critic):
#         super().__init__()
#         self.actor = copy.deepcopy(actor_critic.actor)
#         self.is_recurrent = actor_critic.is_recurrent
#         self.memory = copy.deepcopy(actor_critic.memory.rnn)
#         self.memory.cpu()
#         self.hidden_encoder = copy.deepcopy(actor_critic.hidden_encoder)
#         self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
#         self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

#     def forward(self, x):
#         out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
#         self.hidden_state[:] = h
#         self.cell_state[:] = c
#         latent = self.hidden_encoder(out.squeeze(0))
#         return self.actor(torch.cat((x, latent), dim=1))

#     @torch.jit.export
#     def reset_memory(self):
#         self.hidden_state[:] = 0.
#         self.cell_state[:] = 0.

#     def export(self, path):
#         os.makedirs(path, exist_ok=True)
#         path = os.path.join(path, 'policy_lstm.pt')
#         self.to('cpu')
#         traced_script_module = torch.jit.script(self)
#         traced_script_module.save(path)

class PolicyExporterHIM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)

    def forward(self, obs_history):
        parts = self.estimator(obs_history)[:, 0:19]
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        return self.actor(torch.cat((obs_history[:, 0:45], vel, z), dim=1))

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
    
    
