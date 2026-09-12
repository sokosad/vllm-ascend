# DSV4 v0.26 Policy4

仅适用于本轮 DeepSeek-V4-Flash-0731-w8a8 的 v0.26 环境。
补丁基于 vllm-ascend 提交 `85625bd772f27a4a13bc8e548f131af5ce3ff734`。

## 应用补丁

在容器实际使用的 `/vllm-workspace/vllm-ascend` 代码目录执行，替换补丁绝对路径：

```bash
cd /vllm-workspace/vllm-ascend
git apply --check /path/to/v26/0001-policy4-dsv4.patch
git apply /path/to/v26/0001-policy4-dsv4.patch
```

在服务停止时应用；检查不通过不要强行覆盖。保留原镜像的 vLLM、CANN、torch-npu 和编译算子。
新启动进程必须加载此目录，而不是旧实验 runtime 的 PYTHONPATH。

## 使能方式（本轮实测配置）

只在 P 节点启用，D 节点不开 EPLB。保持原启动脚本其他配置不变。

```bash
export DYNAMIC_EPLB=True
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
```

P 节点保留 `--enable-expert-parallel --enforce-eager`。
将原 `--additional-config` 中的 `eplb_config` 替换为以下配置，不要删除其他成员：

```json
{
  "dynamic_eplb": true,
  "eplb_policy_type": 4,
  "eplb_node_role": "prefill",
  "eplb_heat_collection_stage": "prefill",
  "expert_heat_collection_interval": 400,
  "algorithm_execution_interval": 50,
  "num_redundant_experts": 22
}
```

本轮 P 的完整 additional-config：

```bash
--additional-config '{"enable_cpu_binding":true,"enable_dsa_cp":true,"enable_flashcomm1":true,"eplb_config":{"dynamic_eplb":true,"eplb_policy_type":4,"eplb_node_role":"prefill","eplb_heat_collection_stage":"prefill","expert_heat_collection_interval":400,"algorithm_execution_interval":50,"num_redundant_experts":22}}'
```

P 为 DP4/TP4/EP16；K22 表示每卡跨层共享 22 个冗余槽，共 352 个，不是每层 22 个。
本轮保持原脚本的 DSpark 5 token、draft eager、batched tokens 8192、max seqs 64 和 prefix cache 开启。
模型、KV 连接、地址端口、proxy、数据集与 bench 均沿用原脚本，不需要新增部署脚本。

```bash
grep -E 'Apply plan|Migration completed' service.log
```

补丁保留 P/D 两种 Policy4 实现，但本说明只提供此次验证过的 P 节点配置。
全局池启动阶段使用 `min(expert_heat_collection_interval, 200)` 收集热度；首次迁移完成后，在周期边界恢复配置的 400。稳定窗口与迁移门槛不变，不需要新增配置。

2026-09-12 启动窗口优化运行完成 8000 请求，成功 8000、失败 0；首次迁移后已观察到恢复 400 的日志。该实验 runtime 含诊断采集，发布补丁不包含固定路径的 NPZ/JSONL 诊断写入，不能视为发布包重新完成了性能或精度验证。
本补丁不包含全局池 FULL_DECODE_ONLY 适配，该项尚待单独验证。
