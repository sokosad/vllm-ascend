# SPDX-License-Identifier: Apache-2.0
"""Dynamic CRAFT placement for the fixed-capacity EPLB runtime."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vllm_ascend import envs

from .policy_abstract import EplbPolicy
from .policy_craft import build_craft_placement, replay_balancedness


_MIN_GAIN_ENV = "VLLM_ASCEND_DYNAMIC_CRAFT_MIN_GAIN"
_MAX_APPLIES_ENV = "VLLM_ASCEND_DYNAMIC_CRAFT_MAX_APPLIES"
_PER_LAYER_GAIN_EPSILON = 1e-12
_MAX_MIGRATION_LAYER_FRACTION = 0.25


@dataclass(frozen=True)
class DynamicCraftDecision:
    should_apply: bool
    reason: str
    placement: tuple[tuple[tuple[int, ...], ...], ...]
    priority: tuple[int, ...]
    current_balancedness: float
    predicted_balancedness: float
    gain: float
    changed_slots: int
    stable_windows: int


def _env_min_gain() -> float:
    value = float(envs.VLLM_ASCEND_DYNAMIC_CRAFT_MIN_GAIN)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{_MIN_GAIN_ENV} must satisfy 0 < value < 1")
    return value


def _env_max_applies() -> int | None:
    value = envs.VLLM_ASCEND_DYNAMIC_CRAFT_MAX_APPLIES
    if value is None:
        return None
    if value < 0:
        raise ValueError(f"{_MAX_APPLIES_ENV} must be a non-negative integer")
    return int(value)


def _numpy(value, dtype) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _inputs(current_expert_table, expert_workload) -> tuple[np.ndarray, np.ndarray, int]:
    table = _numpy(current_expert_table, np.int64).copy()
    load = _numpy(expert_workload, np.float64).copy()
    if table.ndim != 3 or table.shape != load.shape or 0 in table.shape:
        raise ValueError("Policy4 expects matching non-empty [L, R, S] table and load")
    if np.any(table < 0) or not np.all(np.isfinite(load)) or np.any(load < 0):
        raise ValueError("Policy4 table and load must be non-negative and finite")
    num_experts = int(table.max()) + 1
    expected = np.arange(num_experts, dtype=np.int64)
    for layer in table:
        if not np.array_equal(np.unique(layer), expected):
            raise ValueError("Policy4 placement must cover every logical expert")
        for rank in layer:
            if np.unique(rank).size != rank.size:
                raise ValueError("Policy4 cannot place two copies on one rank")
    return table, load, num_experts


def _logical_heat(table: np.ndarray, load: np.ndarray, num_experts: int) -> np.ndarray:
    heat = np.zeros((table.shape[0], num_experts), dtype=np.float64)
    for layer in range(table.shape[0]):
        np.add.at(heat[layer], table[layer].reshape(-1), load[layer].reshape(-1))
    return heat


def _placement_maps(table: np.ndarray, num_experts: int):
    phy2log = []
    phy2rank = []
    copies = np.empty((table.shape[0], num_experts), dtype=np.int64)
    for layer in range(table.shape[0]):
        logical = table[layer].reshape(-1)
        phy2log.append(logical)
        phy2rank.append(np.repeat(np.arange(table.shape[1]), table.shape[2]))
        copies[layer] = np.bincount(logical, minlength=num_experts)
    return tuple(phy2log), tuple(phy2rank), copies


def _placement_risk_by_layer(
    current: np.ndarray,
    candidate: np.ndarray,
    heat: np.ndarray,
    num_experts: int,
) -> np.ndarray:
    """Estimate migration and communication-locality risk per layer.

    The runtime currently exposes logical-expert heat, but not a
    source-rank-to-expert traffic matrix.  Retaining the current owners of hot
    experts is therefore the safest available locality proxy.  Peak per-rank
    slot churn additionally avoids concentrating weight transfers on one rank.
    """
    changed = candidate != current
    mean_rank_churn = changed.mean(axis=(1, 2))
    peak_rank_churn = changed.mean(axis=2).max(axis=1)

    current_owners = np.zeros(
        (current.shape[0], num_experts, current.shape[1]), dtype=bool
    )
    candidate_owners = np.zeros_like(current_owners)
    for layer in range(current.shape[0]):
        for rank in range(current.shape[1]):
            current_owners[layer, current[layer, rank], rank] = True
            candidate_owners[layer, candidate[layer, rank], rank] = True

    current_copies = current_owners.sum(axis=2)
    retired_owner_fraction = np.divide(
        np.count_nonzero(current_owners & ~candidate_owners, axis=2),
        current_copies,
        out=np.zeros_like(current_copies, dtype=np.float64),
        where=current_copies != 0,
    )
    total_heat = heat.sum(axis=1)
    hot_owner_churn = np.divide(
        (heat * retired_owner_fraction).sum(axis=1),
        total_heat,
        out=np.zeros_like(total_heat, dtype=np.float64),
        where=total_heat != 0,
    )

    return (mean_rank_churn + peak_rank_churn + hot_owner_churn) / 3.0


def _align_retained_slots(current: np.ndarray, desired) -> np.ndarray:
    result = np.full_like(current, -1)
    for layer in range(current.shape[0]):
        for rank in range(current.shape[1]):
            wanted = [int(item) for item in desired[layer][rank]]
            wanted_set = set(wanted)
            retained = set()
            for slot, expert in enumerate(current[layer, rank]):
                if int(expert) in wanted_set:
                    result[layer, rank, slot] = expert
                    retained.add(int(expert))
            remaining = (item for item in wanted if item not in retained)
            for slot in np.flatnonzero(result[layer, rank] < 0):
                result[layer, rank, slot] = next(remaining)
    return result


def _immutable(table: np.ndarray):
    return tuple(
        tuple(tuple(int(value) for value in rank) for rank in layer)
        for layer in table
    )


class DynamicCraftPlanner:
    def __init__(
        self,
        stable_windows: int = 2,
        cooldown_windows: int = 1,
        amortization_horizon: float = 4.0,
        migration_cost: float = 0.02,
        min_gain: float | None = None,
        max_applies: int | None = None,
        num_redundant_experts: int | None = None,
    ) -> None:
        if stable_windows < 1 or cooldown_windows < 0:
            raise ValueError("invalid Policy4 stability settings")
        self.min_gain = _env_min_gain() if min_gain is None else float(min_gain)
        self.max_applies = (
            _env_max_applies() if max_applies is None else int(max_applies)
        )
        if not 0 < self.min_gain < 1:
            raise ValueError("min_gain must be in (0, 1)")
        if self.max_applies is not None and self.max_applies < 0:
            raise ValueError("max_applies must be non-negative")
        self.num_redundant_experts = num_redund_experts = (
            None
            if num_redundant_experts is None
            else int(num_redundant_experts)
        )
        if num_redund_experts is not None and num_redund_experts <= 0:
            raise ValueError("num_redundant_experts must be positive")
        self.stable_windows = stable_windows
        self.cooldown_windows = cooldown_windows
        self.amortization_horizon = float(amortization_horizon)
        self.migration_cost = float(migration_cost)
        self.window = 0
        self.apply_count = 0
        self.last_apply_window: int | None = None
        self.pending: np.ndarray | None = None
        self.pending_base: np.ndarray | None = None
        self.pending_streak = 0

    @staticmethod
    def _scores(table: np.ndarray, heat: np.ndarray, num_experts: int) -> np.ndarray:
        return replay_balancedness(
            heat, *_placement_maps(table, num_experts), table.shape[1]
        )

    @classmethod
    def _limit_migration_scope(
        cls,
        current: np.ndarray,
        candidate: np.ndarray,
        current_score: np.ndarray,
        heat: np.ndarray,
        num_experts: int,
        amortization_horizon: float,
        migration_cost: float,
    ) -> np.ndarray:
        changed_by_layer = np.count_nonzero(candidate != current, axis=(1, 2))
        changed_layers = np.flatnonzero(changed_by_layer)
        if changed_layers.size == 0:
            return candidate

        gain_by_layer = cls._scores(candidate, heat, num_experts) - current_score
        risk_by_layer = _placement_risk_by_layer(
            current, candidate, heat, num_experts
        )
        net_gain_by_layer = (
            gain_by_layer * amortization_horizon
            - risk_by_layer * migration_cost
        )
        profitable_layers = changed_layers[
            net_gain_by_layer[changed_layers] > _PER_LAYER_GAIN_EPSILON
        ]
        if profitable_layers.size == 0:
            return current.copy()

        max_layers = max(
            1,
            int(np.ceil(current.shape[0] * _MAX_MIGRATION_LAYER_FRACTION)),
        )
        selected_layers = sorted(
            profitable_layers.tolist(),
            key=lambda layer: (
                -float(net_gain_by_layer[layer]),
                -float(
                    gain_by_layer[layer]
                    / max(risk_by_layer[layer], _PER_LAYER_GAIN_EPSILON)
                ),
                -float(gain_by_layer[layer]),
                layer,
            ),
        )[:max_layers]
        limited = current.copy()
        limited[selected_layers] = candidate[selected_layers]
        return limited

    def _eligible(
        self,
        current: np.ndarray,
        candidate: np.ndarray,
        current_score: np.ndarray,
        heat: np.ndarray,
        num_experts: int,
    ) -> tuple[bool, np.ndarray, float, int]:
        predicted = self._scores(candidate, heat, num_experts)
        per_layer_gain = predicted - current_score
        gain = float(per_layer_gain.mean())
        changed = int(np.count_nonzero(candidate != current))
        placement_risk = _placement_risk_by_layer(
            current, candidate, heat, num_experts
        )
        cost = float(placement_risk.mean()) * self.migration_cost
        ready = (
            changed > 0
            and gain >= self.min_gain
            and float(per_layer_gain.min()) >= -_PER_LAYER_GAIN_EPSILON
            and gain * self.amortization_horizon > cost
        )
        return ready, per_layer_gain, float(predicted.mean()), changed

    def plan(self, current_expert_table, expert_workload) -> DynamicCraftDecision:
        table, load, num_experts = _inputs(current_expert_table, expert_workload)
        self.window += 1
        heat = _logical_heat(table, load, num_experts)
        current_by_layer = self._scores(table, heat, num_experts)
        replicas = table.shape[1] * table.shape[2] - num_experts
        if (
            self.num_redundant_experts is not None
            and replicas != self.num_redundant_experts
        ):
            raise ValueError(
                "Policy4 runtime placement has "
                f"{replicas} redundant experts, expected "
                f"{self.num_redundant_experts} from configuration"
            )
        reason = "awaiting-stable-window"

        if replicas <= 0:
            candidate = table
            reason = "no-redundant-capacity"
        else:
            _, desired, _, _, _ = build_craft_placement(
                heat, np.full(table.shape[0], replicas), table.shape[1]
            )
            candidate = _align_retained_slots(table, desired)
            candidate = self._limit_migration_scope(
                table,
                candidate,
                current_by_layer,
                heat,
                num_experts,
                self.amortization_horizon,
                self.migration_cost,
            )

        fresh_ready, fresh_gain_by_layer, fresh_score, fresh_changed = self._eligible(
            table, candidate, current_by_layer, heat, num_experts
        )
        selected = candidate
        selected_ready = fresh_ready
        selected_gain_by_layer = fresh_gain_by_layer
        selected_score = fresh_score
        selected_changed = fresh_changed

        base_unchanged = (
            self.pending_base is not None
            and np.array_equal(self.pending_base, table)
        )
        if not fresh_ready:
            self.pending = None
            self.pending_base = None
            self.pending_streak = 0
            reason = "insufficient-gain"
        elif self.pending is None or not base_unchanged:
            self.pending = candidate.copy()
            self.pending_base = table.copy()
            self.pending_streak = 1
        elif np.array_equal(self.pending, candidate):
            self.pending_streak += 1
            selected = self.pending
        else:
            pending_ready, pending_gains, pending_score, pending_changed = (
                self._eligible(
                    table, self.pending, current_by_layer, heat, num_experts
                )
            )
            # Stability means that the incumbent remains beneficial under a
            # fresh load window, not that a deterministic optimizer reproduces
            # the same placement. Chasing every newly optimal placement keeps
            # resetting the streak on otherwise stable traffic and can prevent
            # EPLB from ever applying a valid plan.
            if pending_ready:
                self.pending_streak += 1
                selected = self.pending
                selected_ready = True
                selected_gain_by_layer = pending_gains
                selected_score = pending_score
                selected_changed = pending_changed
            else:
                self.pending = candidate.copy()
                self.pending_base = table.copy()
                self.pending_streak = 1

        cooldown = (
            self.last_apply_window is not None
            and self.window - self.last_apply_window <= self.cooldown_windows
        )
        capped = (
            self.max_applies is not None and self.apply_count >= self.max_applies
        )
        should_apply = (
            selected_ready
            and self.pending_streak >= self.stable_windows
            and not cooldown
            and not capped
        )
        stable_windows = self.pending_streak
        if should_apply:
            reason = "apply"
            self.apply_count += 1
            self.last_apply_window = self.window
            self.pending = None
            self.pending_base = None
            self.pending_streak = 0
        elif cooldown:
            reason = "cooldown"
        elif capped:
            reason = "max-applies"

        selected_risk_by_layer = _placement_risk_by_layer(
            table, selected, heat, num_experts
        )
        selected_net_gain_by_layer = (
            selected_gain_by_layer * self.amortization_horizon
            - selected_risk_by_layer * self.migration_cost
        )
        priority = tuple(
            int(value)
            for value in np.argsort(-selected_net_gain_by_layer, kind="stable")
        )
        return DynamicCraftDecision(
            should_apply=should_apply,
            reason=reason,
            placement=_immutable(selected if should_apply else table),
            priority=priority,
            current_balancedness=float(current_by_layer.mean()),
            predicted_balancedness=selected_score,
            gain=float(selected_gain_by_layer.mean()),
            changed_slots=selected_changed,
            stable_windows=stable_windows,
        )


class DynamicCraftEplb(EplbPolicy):
    def __init__(self, **planner_config) -> None:
        self.planner = DynamicCraftPlanner(**planner_config)
        self.last_decision: DynamicCraftDecision | None = None

    def rebalance_experts(self, current_expert_table, expert_workload):
        self.last_decision = self.planner.plan(
            current_expert_table, expert_workload
        )
        decision = self.last_decision
        return (
            int(decision.should_apply),
            np.asarray(decision.priority),
            decision.placement,
        )


__all__ = ["DynamicCraftDecision", "DynamicCraftEplb", "DynamicCraftPlanner"]
