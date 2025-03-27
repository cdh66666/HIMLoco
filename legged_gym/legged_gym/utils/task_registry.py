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
from datetime import datetime
from typing import Tuple
import torch
import numpy as np

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner, HIMOnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# 定义一个任务注册表类，用于管理任务类、环境配置和训练配置
class TaskRegistry():
    def __init__(self):
        """
        初始化任务注册表。

        此方法用于创建三个空字典，分别用于存储任务类、环境配置和训练配置。
        这些字典将在后续的任务注册和管理中发挥重要作用。
        """
        # 存储任务名称到任务类的映射
        self.task_classes = {}
        # 存储任务名称到环境配置的映射
        self.env_cfgs = {}
        # 存储任务名称到训练配置的映射
        self.train_cfgs = {}
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        """
        向注册表中注册一个新的任务。

        参数:
        name (str): 任务的名称。
        task_class (VecEnv): 任务类，继承自 VecEnv。
        env_cfg (LeggedRobotCfg): 环境配置，基于 LeggedRobotCfg。
        train_cfg (LeggedRobotCfgPPO): 训练配置，基于 LeggedRobotCfgPPO。
        """
        # 将任务类添加到任务类字典中
        self.task_classes[name] = task_class
        # 将环境配置添加到环境配置字典中
        self.env_cfgs[name] = env_cfg
        # 将训练配置添加到训练配置字典中
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        """
        根据任务名称获取对应的任务类。

        参数:
        name (str): 任务的名称。

        返回:
        VecEnv: 对应的任务类。
        """
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        """
        根据任务名称获取对应的环境配置和训练配置。

        参数:
        name (str): 任务的名称。

        返回:
        Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]: 一个元组，包含环境配置和训练配置。
        """
        # 从训练配置字典中获取指定任务名称对应的训练配置
        train_cfg = self.train_cfgs[name]
        # 从环境配置字典中获取指定任务名称对应的环境配置
        env_cfg = self.env_cfgs[name]
        # 将训练配置中的随机种子复制到环境配置中，确保两者使用相同的随机种子
        env_cfg.seed = train_cfg.seed
        # 返回包含环境配置和训练配置的元组
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """
        从已注册的名称或提供的配置文件中创建一个环境。

        参数:
        name (string): 已注册环境的名称。
        args (Args, 可选): Isaac Gym 命令行参数。如果为 None，则会调用 get_args()。默认为 None。
        env_cfg (Dict, 可选): 用于覆盖已注册配置的环境配置文件。默认为 None。

        异常:
        ValueError: 如果没有与 'name' 对应的已注册环境，则抛出此错误。

        返回:
        isaacgym.VecTaskPython: 创建的环境
        Dict: 对应的配置文件
        """
        # 如果没有传入参数，则获取命令行参数
        if args is None:
            args = get_args()
        # 检查是否有使用该名称注册的环境
        if name in self.task_classes:
            # 从注册表中获取对应的任务类
            task_class = self.get_task_class(name)
        else:
            # 如果没有注册该任务，则抛出异常
            raise ValueError(f"Task with name: {name} was not registered")
        # 如果没有提供环境配置，则从注册表中加载配置
        if env_cfg is None:
            # 加载环境配置和训练配置，这里只使用环境配置
            env_cfg, _ = self.get_cfgs(name)
        # 根据命令行参数更新环境配置（如果指定了）
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        # 设置随机种子以确保结果可复现
        set_seed(env_cfg.seed)

        # 解析模拟参数，先将环境配置中的模拟部分转换为字典
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        # 根据命令行参数解析模拟参数
        sim_params = parse_sim_params(args, sim_params)
        # 使用任务类创建环境实例
        env = task_class(
            cfg=env_cfg,  # 传入环境配置
            sim_params=sim_params,  # 传入模拟参数
            physics_engine=args.physics_engine,  # 传入物理引擎类型
            sim_device=args.sim_device,  # 传入模拟设备
            headless=args.headless  # 传入是否无头模式
        )
        # 返回创建的环境实例和对应的环境配置
        return env, env_cfg


    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """
        从已注册的名称或提供的配置文件中创建训练算法。

        参数:
            env (isaacgym.VecTaskPython): 用于训练的环境（TODO: 从算法内部移除）
            name (string, 可选): 已注册环境的名称。如果为 None，则将使用配置文件。默认为 None。
            args (Args, 可选): Isaac Gym 命令行参数。如果为 None，则会调用 get_args()。默认为 None。
            train_cfg (Dict, 可选): 训练配置文件。如果为 None，则将使用 'name' 来获取配置文件。默认为 None。
            log_root (str, 可选): Tensorboard 的日志目录。设置为 'None' 以避免日志记录（例如在测试时）。
                                    日志将保存在 <log_root>/<date_time>_<run_name> 中。默认为 "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>。

        异常:
            ValueError: 如果既没有提供 'name' 也没有提供 'train_cfg'，则抛出此错误。
            Warning: 如果同时提供了 'name' 和 'train_cfg'，则将忽略 'name'。

        返回:
            PPO: 创建的算法
            Dict: 对应的配置文件
        """
        # 如果没有传入参数，则获取命令行参数
        if args is None:
            args = get_args()
        # 如果传入了配置文件，则使用它们；否则，从名称加载
        if train_cfg is None:
            # 如果 train_cfg 为 None，检查 name 是否为 None
            if name is None:
                # 若 name 也为 None，抛出异常，因为必须提供至少一个
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # 加载配置文件
            # 若 name 不为 None，根据 name 从注册表中获取环境配置和训练配置，这里只使用训练配置
            _, train_cfg = self.get_cfgs(name)
        else:
            # 若 train_cfg 不为 None，检查 name 是否不为 None
            if name is not None:
                # 若 name 不为 None，打印警告信息，表明 'train_cfg' 已提供，将忽略 'name'
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # 根据命令行参数更新训练配置（如果指定了）
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        # 根据 log_root 的值设置日志目录
        if log_root == "default":
            # 如果 log_root 为 "default"，将日志根目录设置为 <path_to_LEGGED_GYM>/logs/<experiment_name>
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            # 生成具体的日志目录，格式为 <log_root>/<date_time>_<run_name>
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            # 如果 log_root 为 None，不设置日志目录
            log_dir = None
        else:
            # 否则，将日志目录设置为 <log_root>/<date_time>_<run_name>
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        
        # 将训练配置类转换为字典
        train_cfg_dict = class_to_dict(train_cfg)
        # 使用环境、训练配置字典、日志目录和指定设备创建一个训练算法运行器实例
        runner = HIMOnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)

        # 保存是否恢复训练的标志
        resume = train_cfg.runner.resume
        if resume:
            # 如果需要恢复训练
            # 获取之前训练的模型加载路径
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            # 打印加载模型的信息
            print(f"Loading model from: {resume_path}")
            # 从指定路径加载模型
            runner.load(resume_path)
        # 返回创建的训练算法运行器实例和对应的训练配置
        return runner, train_cfg

# make global task registry
task_registry = TaskRegistry()