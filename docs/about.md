# 项目与作者

<p class="institution-mark">
  <a href="https://www.ustc.edu.cn/">
    <img src="../assets/ustc-logo.png" alt="University of Science and Technology of China">
  </a>
</p>

Kunlunxin-FFT 是 Teng Wang 维护的开源研究软件，研究工作开展于 High Efficient Intelligent
Computing Lab，并以中国科学技术大学苏州高等研究院为作者机构归属。

本项目将 [cuButterfly](https://github.com/TruNcat3/cuButterfly) 的混合空间-时间蝶形计算方法
迁移到 Kunlunxin P800，并独立实现 Triton/XMLIR/XPU kernel、硬件 lowering、冻结 selector、
正确性门禁和实验归档。方法迁移对象是计算流编排，不是 CUDA kernel 的逐行翻译。

关于 Roofline、阶段依赖、阶段/数据双维空间时间展开、流与核心的正交分层等观点和详细分析，来源于
并应归属于 cuButterfly。Kunlunxin-FFT 的贡献是 P800 平台实践和证据。引用方法论时请同时引用
cuButterfly；引用 P800 实现/数据时引用本仓库。

## 作者与联系

| 项目 | 信息 |
|---|---|
| 作者 / 维护者 | **Teng Wang** |
| 实验室 | High Efficient Intelligent Computing Lab |
| 机构 | [Suzhou Institute for Advanced Research, University of Science and Technology of China](https://sz.ustc.edu.cn/en/index.html) |
| 所在地 | Suzhou, China |
| 邮箱 | [wangt635@ustc.edu.cn](mailto:wangt635@ustc.edu.cn) |
| GitHub | [TruNcat3](https://github.com/TruNcat3) |

## 引用

机器可读元数据见仓库根目录
[`CITATION.cff`](https://github.com/TruNcat3/kunlunxin-fft/blob/master/CITATION.cff)。

```bibtex
@software{wang2026kunlunxinfft,
  author       = {Teng Wang},
  title        = {Kunlunxin-FFT: A Hardware-Mapped FFT Operator Library for Kunlunxin P800},
  year         = {2026},
  url          = {https://github.com/TruNcat3/kunlunxin-fft},
  organization = {High Efficient Intelligent Computing Lab,
                  Suzhou Institute for Advanced Research,
                  University of Science and Technology of China}
}
```

## 许可与机构声明

代码和仓库文档采用 Apache-2.0，允许修改与发布，并须保留许可证和通知。机构名称/标识仅用于说明
作者研究归属；本项目不是中国科学技术大学或苏州高等研究院的官方软件发布，也不表示机构认证或
背书。中科大标识不属于 Apache-2.0 软件授权范围，详见
[`THIRD_PARTY_NOTICES.md`](https://github.com/TruNcat3/kunlunxin-fft/blob/master/THIRD_PARTY_NOTICES.md)。

返回：[阅读入口](index.md)。
