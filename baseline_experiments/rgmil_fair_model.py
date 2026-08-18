"""RGMIL core model with reproducibility and evaluation bug fixes."""

from __future__ import annotations

import random
from collections import namedtuple
from copy import deepcopy
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from torch_geometric.nn import GATConv


Transition = namedtuple("Transition", ["states", "actions", "reward", "next_states"])


class Memory:
    def __init__(self, memory_size: int, batch_size: int):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory: List[Transition] = []

    def save(self, states, actions, reward, next_states) -> None:
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        self.memory.append(Transition(states, actions, reward, next_states))

    def sample(self):
        return random.sample(self.memory, self.batch_size)


class Policy(nn.Module):
    def __init__(self, state_dim: int, action_count: int, slope: float, layer_count: int):
        super().__init__()
        self.slope = slope
        self.layers = nn.ModuleList([nn.Linear(state_dim, state_dim) for _ in range(layer_count)])
        self.classifier = nn.Linear(state_dim, action_count)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            observation = F.leaky_relu(layer(observation), negative_slope=self.slope)
        return F.leaky_relu(self.classifier(observation), negative_slope=self.slope)


class Agent(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_space: Sequence[float],
        slope: float,
        learning_rate: float,
        weight_decay: float,
        layer_count: int,
    ):
        super().__init__()
        self.space = np.asarray(action_space)
        self.space_prob = np.flipud(self.space / np.sum(self.space))
        self.policy = Policy(state_dim, len(action_space), slope, layer_count)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def forward(self, observation: torch.Tensor, epsilon: float):
        q_values = F.softmax(self.policy(observation), dim=0)
        if random.random() <= epsilon:
            action = int(np.random.choice(np.arange(len(self.space)), p=self.space_prob))
        else:
            action = int(torch.argmax(q_values).item())
        return q_values, action


class GAT(nn.Module):
    """Paper GAT + attention pooling; source-dataset graph edges are never used."""

    def __init__(
        self,
        data_name: str,
        feat_dim: int,
        max_layer_num: int,
        drop_rate: float,
        slope_rate: float,
        device: torch.device,
    ):
        super().__init__()
        self.data_name = data_name
        self.drop_rate = drop_rate
        self.slope_rate = slope_rate
        self.device = device
        self.gat_layers = nn.ModuleList([GATConv(feat_dim, feat_dim) for _ in range(max_layer_num)])
        self.gat_w = nn.Linear(feat_dim, feat_dim)
        self.gat_vec = nn.Linear(feat_dim, 1, bias=False)
        self.transformer = nn.Linear(feat_dim, max(1, feat_dim // 2))
        self.classifier = nn.Linear(max(1, feat_dim // 2), 1)
        self.loss_function = nn.BCEWithLogitsLoss()

    @staticmethod
    def build_edges(similarity: np.ndarray, threshold: float) -> np.ndarray:
        # Copy is essential: mutating cached similarities makes later actions invalid.
        binary = np.asarray(similarity >= threshold, dtype=np.float32)
        edges = coo_matrix(binary)
        return np.vstack((edges.row, edges.col)).astype(np.int64)

    def forward(self, inputs):
        ins_feats, similarity, threshold, layer_num = inputs
        edges = self.build_edges(similarity, float(threshold))
        features = torch.as_tensor(ins_feats, dtype=torch.float32, device=self.device)
        residual = features
        edge_index = torch.as_tensor(edges, dtype=torch.long, device=self.device)
        for layer_idx in range(int(layer_num)):
            features = self.gat_layers[layer_idx](features, edge_index)
            if layer_idx > 0 and (layer_idx + 1) % 2 == 0:
                features = features + residual
                residual = features
            features = F.leaky_relu(features, negative_slope=self.slope_rate)
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        coefficients = self.gat_vec(torch.tanh(self.gat_w(features))).reshape(-1)
        coefficients = F.softmax(coefficients, dim=0).unsqueeze(-1)
        bag_features = torch.sum(coefficients * features, dim=0)
        detached_bag_features = bag_features.detach().cpu().numpy()
        bag_features = F.leaky_relu(self.transformer(bag_features), negative_slope=self.slope_rate)
        # Raw logits are required by BCEWithLogitsLoss; the original public code applied
        # LeakyReLU here, which restricts the negative logit range.
        bag_logits = self.classifier(bag_features).reshape(1)
        return bag_logits, detached_bag_features


class RGMILSearch(nn.Module):
    """Two-agent VDN action search using exact train blocks and a fixed validation block."""

    def __init__(
        self,
        data_name: str,
        threshold_space: Sequence[float],
        layer_space: Sequence[int],
        state_1_dim: int,
        state_2_dim: int,
        gnn_learning_rate: float,
        gnn_weight_decay: float,
        agent_learning_rate: float,
        agent_weight_decay: float,
        policy_layer_num: int,
        drop_rate: float,
        slope_rate: float,
        discount_rate: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_steps: int,
        history_num: int,
        reward_tolerance: float,
        memory_size: int,
        memory_batch_size: int,
        device: torch.device,
    ):
        super().__init__()
        self.threshold_space = list(threshold_space)
        self.layer_space = list(layer_space)
        self.discount_rate = discount_rate
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        self.history_num = history_num
        self.reward_tolerance = reward_tolerance
        self.device = device
        joint_state_dim = state_1_dim + state_2_dim
        self.gnn = GAT(data_name, state_2_dim, max(layer_space), drop_rate, slope_rate, device).to(device)
        self.gnn_optimizer = torch.optim.Adam(
            self.gnn.parameters(), lr=gnn_learning_rate, weight_decay=gnn_weight_decay
        )
        self.agent_1 = Agent(
            joint_state_dim, threshold_space, slope_rate, agent_learning_rate,
            agent_weight_decay, policy_layer_num
        ).to(device)
        self.agent_2 = Agent(
            joint_state_dim, layer_space, slope_rate, agent_learning_rate,
            agent_weight_decay, policy_layer_num
        ).to(device)
        self.target_1 = deepcopy(self.agent_1).eval()
        self.target_2 = deepcopy(self.agent_2).eval()
        self.memory = Memory(memory_size, memory_batch_size)
        self.memory_batch_size = memory_batch_size
        self.mse = nn.MSELoss()
        self.history_performance = [0.0] * history_num
        self.reward_trace = [0.0] * history_num
        self.loss_gnn_trace: List[float] = []
        self.loss_agent_trace: List[float] = [0.0] * history_num
        self.threshold_trace: List[float] = []
        self.layer_trace: List[int] = []
        self.combination_count = {}
        self.copy_step = 0
        self.current_block = 0
        self.threshold = float(threshold_space[0])
        self.layer_num = int(layer_space[0])

    @staticmethod
    def _bags(block):
        return block[:-1]

    @staticmethod
    def _state(block, device):
        state_1, state_2 = block[-1]
        return (
            torch.as_tensor(state_1, dtype=torch.float32, device=device),
            torch.as_tensor(state_2, dtype=torch.float32, device=device),
        )

    @staticmethod
    def _joint_observations(state_1, state_2):
        return torch.cat((state_1, state_2), dim=0), torch.cat((state_2, state_1), dim=0)

    def _train_gnn(self, blocks) -> float:
        self.gnn.train()
        loss_sum = 0.0
        count = 0
        for block in blocks[:-1]:
            for label, features, similarity, _ in self._bags(block):
                target = torch.tensor([label], dtype=torch.float32, device=self.device)
                self.gnn_optimizer.zero_grad(set_to_none=True)
                logits, _ = self.gnn((features, similarity, self.threshold, self.layer_num))
                loss = self.gnn.loss_function(logits, target)
                loss.backward()
                self.gnn_optimizer.step()
                loss_sum += float(loss.detach())
                count += 1
        value = loss_sum / max(count, 1)
        self.loss_gnn_trace.append(value)
        return value

    @torch.inference_mode()
    def _validation_accuracy(self, validation_block) -> float:
        self.gnn.eval()
        correct = 0
        bags = self._bags(validation_block)
        for label, features, similarity, _ in bags:
            logits, _ = self.gnn((features, similarity, self.threshold, self.layer_num))
            prediction = int(torch.sigmoid(logits).item() > 0.5)
            correct += int(prediction == int(label))
        return correct / max(len(bags), 1)

    def _train_agents(self) -> None:
        transitions = self.memory.sample()
        for transition in transitions:
            obs_1, obs_2 = transition.states
            action_1, action_2 = transition.actions
            q_1, _ = self.agent_1(obs_1, 1e-10)
            q_2, _ = self.agent_2(obs_2, 1e-10)
            with torch.no_grad():
                next_obs_1, next_obs_2 = transition.next_states
                _, next_action_1 = self.agent_1(next_obs_1, 1e-10)
                _, next_action_2 = self.agent_2(next_obs_2, 1e-10)
                target_q_1, _ = self.target_1(next_obs_1, 1e-10)
                target_q_2, _ = self.target_2(next_obs_2, 1e-10)
                target = transition.reward + self.discount_rate * (
                    target_q_1[next_action_1] + target_q_2[next_action_2]
                )
            prediction = q_1[action_1] + q_2[action_2]
            self.agent_1.optimizer.zero_grad(set_to_none=True)
            self.agent_2.optimizer.zero_grad(set_to_none=True)
            loss = self.mse(prediction, target)
            loss.backward()
            self.agent_1.optimizer.step()
            self.agent_2.optimizer.step()
            self.loss_agent_trace.append(float(loss.detach()))
        if self.copy_step % self.history_num == 0:
            self.target_1 = deepcopy(self.agent_1).eval()
            self.target_2 = deepcopy(self.agent_2).eval()
        self.copy_step += 1

    def step(self, blocks, timestep: int) -> bool:
        train_block_count = len(blocks) - 1
        epsilon = float(self.epsilons[min(timestep, len(self.epsilons) - 1)])
        state_1, state_2 = self._state(blocks[self.current_block], self.device)
        obs_1, obs_2 = self._joint_observations(state_1, state_2)
        _, action_1 = self.agent_1(obs_1, epsilon)
        _, action_2 = self.agent_2(obs_2, epsilon)
        self.threshold = float(self.threshold_space[action_1])
        self.layer_num = int(self.layer_space[action_2])
        combination = f"{self.threshold:.2f}-{self.layer_num}"
        self.combination_count[combination] = self.combination_count.get(combination, 0) + 1
        if self.combination_count[combination] <= self.history_num:
            self._train_gnn(blocks)
        validation_accuracy = self._validation_accuracy(blocks[-1])
        historical = float(np.mean(self.history_performance[-self.history_num :]))
        reward = validation_accuracy - historical
        self.history_performance.append(validation_accuracy)
        self.reward_trace.append(reward)
        next_block = int(round(self.threshold + self.layer_num)) % train_block_count
        next_state_1, next_state_2 = self._state(blocks[next_block], self.device)
        next_obs = self._joint_observations(next_state_1, next_state_2)
        self.memory.save([obs_1, obs_2], [action_1, action_2], reward, list(next_obs))
        if timestep >= self.history_num and len(self.memory.memory) >= self.memory_batch_size:
            self._train_agents()
        self.threshold_trace.append(self.threshold)
        self.layer_trace.append(self.layer_num)
        self.current_block = next_block
        stable_actions = (
            len(self.threshold_trace) >= self.history_num
            and len(set(self.threshold_trace[-self.history_num :])) == 1
            and len(set(self.layer_trace[-self.history_num :])) == 1
        )
        stable_reward = abs(float(np.mean(self.reward_trace[-self.history_num :]))) <= self.reward_tolerance
        return bool(stable_actions and stable_reward)
