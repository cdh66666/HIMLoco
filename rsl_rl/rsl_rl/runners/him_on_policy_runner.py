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

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO, HIMPPO
from rsl_rl.modules import HIMActorCritic
from rsl_rl.env import VecEnv


class HIMOnPolicyRunner:
    """
    HIMOnPolicyRunner类用于执行基于策略的强化学习训练。

    该类封装了环境、算法和策略，支持训练过程中的数据收集、学习步骤、日志记录和模型保存/加载。

    Attributes:
        cfg (dict): 运行器配置
        alg_cfg (dict): 算法配置
        policy_cfg (dict): 策略配置
        device (str): 设备（CPU或GPU）
        env (VecEnv): 向量环境
        num_actor_obs (int): 演员观察的数量
        num_critic_obs (int): 评论家观察的数量
        alg (HIMPPO): 强化学习算法
        num_steps_per_env (int): 每个环境的步数
        save_interval (int): 保存模型的间隔
        log_dir (str): 日志目录
        writer (SummaryWriter): TensorBoard写入器
        tot_timesteps (int): 总时间步数
        tot_time (float): 总时间
        current_learning_iteration (int): 当前学习迭代次数
    """

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        """
        初始化HIMOnPolicyRunner类。

        Args:
            env (VecEnv): 向量环境
            train_cfg (dict): 训练配置
            log_dir (str, optional): 日志目录. 默认值为 None.
            device (str, optional): 设备（CPU或GPU）. 默认值为 'cpu'.
        """
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        # 如果环境有特权观察，则使用特权观察作为评论家观察；否则使用普通观察
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        self.num_actor_obs = self.env.num_obs
        self.num_critic_obs = num_critic_obs
        # 根据配置中的策略类名获取策略类
        actor_critic_class = eval(self.cfg["policy_class_name"])  # HIMActorCritic
        # 创建策略实例
        actor_critic: HIMActorCritic = actor_critic_class(
            self.env.num_obs,
            num_critic_obs,
            self.env.num_one_step_obs,
            self.env.num_actions,
            **self.policy_cfg
        ).to(self.device)
        # 根据配置中的算法类名获取算法类
        alg_class = eval(self.cfg["algorithm_class_name"])  # HIMPPO
        # 创建算法实例
        self.alg: HIMPPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # 初始化存储和模型
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions]
        )

        # 日志相关设置
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # 重置环境
        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """
        执行学习过程。

        Args:
            num_learning_iterations (int): 学习迭代次数
            init_at_random_ep_len (bool, optional): 是否以随机的情节长度初始化. 默认值为 False.
        """
        # 如果日志目录存在且写入器未初始化，则初始化写入器
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        # 如果需要以随机的情节长度初始化，则随机设置情节长度
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        # 获取环境观察
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        # 如果有特权观察，则使用特权观察作为评论家观察；否则使用普通观察
        critic_obs = privileged_obs if privileged_obs is not None else obs
        # 将观察移动到指定设备
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        # 将策略模型设置为训练模式
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        # 奖励缓冲区，最大长度为100
        rewbuffer = deque(maxlen=100)
        # 情节长度缓冲区，最大长度为100
        lenbuffer = deque(maxlen=100)
        # 当前奖励总和
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # 当前情节长度
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # 总迭代次数
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            # 记录开始时间
            start = time.time()
            # 数据收集阶段
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # 根据观察选择动作
                    actions = self.alg.act(obs, critic_obs)
                    # 执行动作并获取环境反馈
                    obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs = self.env.step(
                        actions
                    )
                    # 如果有特权观察，则使用特权观察作为评论家观察；否则使用普通观察
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    # 将观察、奖励、完成标志等移动到指定设备
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(
                        self.device), dones.to(self.device)
                    termination_ids = termination_ids.to(self.device)
                    termination_privileged_obs = termination_privileged_obs.to(self.device)

                    # 克隆并分离下一个评论家观察
                    next_critic_obs = critic_obs.clone().detach()
                    next_critic_obs[termination_ids] = termination_privileged_obs.clone().detach()

                    # 处理环境步骤
                    self.alg.process_env_step(rewards, dones, infos, next_critic_obs)

                    if self.log_dir is not None:
                        # 记录数据
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        # 获取完成的情节的索引
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        # 将完成情节的奖励添加到奖励缓冲区
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        # 将完成情节的长度添加到情节长度缓冲区
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        # 重置完成情节的奖励总和
                        cur_reward_sum[new_ids] = 0
                        # 重置完成情节的长度
                        cur_episode_length[new_ids] = 0

                # 记录结束时间
                stop = time.time()
                # 计算数据收集时间
                collection_time = stop - start

                # 学习阶段
                start = stop
                # 计算回报
                self.alg.compute_returns(critic_obs)

            # 更新算法参数，获取各种损失
            mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss = self.alg.update()
            # 记录结束时间
            stop = time.time()
            # 计算学习时间
            learn_time = stop - start
            if self.log_dir is not None:
                # 记录日志
                self.log(locals())
            # 如果达到保存间隔，则保存模型
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            # 清空情节信息列表
            ep_infos.clear()

        # 更新当前学习迭代次数
        self.current_learning_iteration += num_learning_iterations
        # 保存最终模型
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        """
        记录训练日志。

        Args:
            locs (dict): 局部变量字典
            width (int, optional): 日志字符串宽度. 默认值为 80.
            pad (int, optional): 日志字符串填充宽度. 默认值为 35.
        """
        # 更新总时间步数
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        # 更新总时间
        self.tot_time += locs['collection_time'] + locs['learn_time']
        # 计算迭代时间
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                # 初始化信息张量
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # 处理标量和零维张量信息
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    # 拼接信息张量
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                # 计算信息张量的平均值
                value = torch.mean(infotensor)
                # 向TensorBoard写入情节信息
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                # 构建情节信息字符串
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        # 计算策略模型动作噪声标准差的平均值
        mean_std = self.alg.actor_critic.std.mean()
        # 计算每秒步数
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        # 向TensorBoard写入各种损失信息
        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/Estimation Loss', locs['mean_estimation_loss'], locs['it'])
        self.writer.add_scalar('Loss/Swap Loss', locs['mean_swap_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        # 向TensorBoard写入策略动作噪声标准差信息
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        # 向TensorBoard写入性能信息
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            # 向TensorBoard写入训练奖励和情节长度信息
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        # 构建学习迭代信息字符串
        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            # 构建包含奖励和情节长度信息的日志字符串
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
                f"""{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        else:
            # 构建不包含奖励和情节长度信息的日志字符串
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
                f"""{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )

        # 添加情节信息到日志字符串
        log_string += ep_string
        # 添加总时间步数、迭代时间、总时间和ETA信息到日志字符串
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        # 打印日志字符串
        print(log_string)

    def save(self, path, infos=None):
        """
        保存模型。

        Args:
            path (str): 保存路径
            infos (dict, optional): 额外信息. 默认值为 None.
        """
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'estimator_optimizer_state_dict': self.alg.actor_critic.estimator.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True):
        """
        加载模型。

        Args:
            path (str): 加载路径
            load_optimizer (bool, optional): 是否加载优化器状态. 默认值为 True.

        Returns:
            dict: 加载的额外信息
        """
        # 加载模型字典
        loaded_dict = torch.load(path)
        # 加载策略模型状态
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            # 加载优化器状态
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            # 加载估计器优化器状态
            self.alg.actor_critic.estimator.optimizer.load_state_dict(loaded_dict['estimator_optimizer_state_dict'])
        # 更新当前学习迭代次数
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        """
        获取推理策略。

        Args:
            device (str, optional): 设备（CPU或GPU）. 默认值为 None.

        Returns:
            function: 推理策略函数
        """
        # 将策略模型设置为评估模式
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            # 将策略模型移动到指定设备
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
