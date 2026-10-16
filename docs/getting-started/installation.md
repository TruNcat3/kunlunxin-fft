# 安装与环境检查

Kunlunxin-FFT 从源码运行，不提供预编译 wheel。设备 kernel 由 Triton 在运行时编译为
XMLIR/XPU 指令；仓库中的 C++ 只用于 host 数学参考与无设备测试。

## 前置条件

| 组件 | 用途 |
|---|---|
| Kunlunxin P800 驱动与可访问设备 | 当前唯一实测硬件 |
| 厂商 xpytorch：`torch`、`torch_xmlir`、`flag_gems` | 设备、runtime 与 context |
| 带 XPU backend 的 Triton | JIT 编译设备 kernel |
| Python 3、NumPy、pytest | API、参考正确性和 host 测试 |
| C++17 编译器 | host 参考程序 |

正式发布数据记录的环境为 `torch 2.9.0+cu129`、XPYTORCH
`XMLIR--bc1b1dc6f-dev+2026032411`、XRE `5.13.0.0`、driver `515.58`。这是一份实验
provenance，不表示其他兼容版本一定失败。

## 获取与检查

```bash
git clone https://github.com/TruNcat3/kunlunxin-fft.git
cd kunlunxin-fft

export CUDA_VISIBLE_DEVICES=6  # 使用服务器分配给你的 P800 设备号
bash scripts/init.sh
bash scripts/build.sh
```

`scripts/init.sh` 只检查 Python、`torch + torch_xmlir`、设备 tensor 和 Triton target；它不会
安装驱动、下载 xpytorch、探测完整硬件 profile 或校准 selector。`scripts/build.sh` 会：

1. 编译并运行 host C++17 参考测试；
2. 默认预热 Triton JIT 缓存，首次执行可能需要数分钟。

仅执行 host 构建与 CI 测试：

```bash
SKIP_TRITON_WARM=1 bash scripts/build.sh
pytest -q
```

## 设备键为什么叫 cuda

参考 xpytorch 栈在导入 `torch_xmlir` 后通过 `torch.device("cuda")` 暴露 P800。这里的 `cuda`
只是兼容设备键，不表示使用 CUDA/PTX 或 NVIDIA GPU；本项目设备代码路径是
`Triton -> XMLIR -> XPU instructions`。

## Profile 边界

`config/device.json` 是提交到仓库的静态 P800 描述。`DeviceProfile.load().hash()` 只能证明调用者
使用了同一份配置，不能验证当前实际驱动、设备或频率与发布环境一致。因此硬件 probe 与环境绑定
**已落地**：在设备主机上运行 `scripts/hw_probe.py` 会采集实时指纹（driver/torch/triton/XRE/设备
属性）并写入 env 绑定的 `config/calibration.json`；selector 表在建表时从测量 manifest 绑定同一
指纹。运行时 `select()` 与校准加载都会核对实时指纹，缺失、未知或失配一律 fail closed 到 M0
（未绑定 env 的旧表只允许显式离线检查，不参与运行时选择）。安装期**全自动**探测与校准仍是
[未来计划](../roadmap.md)；当前流程是手动运行 `hw_probe.py`。

下一页：[首次运行](quickstart.md)。
