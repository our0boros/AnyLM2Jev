<div align="center">

```
▄████▄ ▄▄  ▄▄ ▄▄ ▄▄ ██     ██▄  ▄██ ████▄    ██ ▄▄▄▄▄ ▄▄ ▄▄
██▄▄██ ███▄██ ▀███▀ ██     ██ ▀▀ ██  ▄██▀    ██ ██▄▄  ██▄██
██  ██ ██ ▀██   █   ██████ ██    ██ ███▄▄ ████▀ ██▄▄▄  ▀█▀
```

**把任意小模型变成 Jev 式的决策模型。**
类型化决策、单次前向、可校准置信度、与选项顺序无关的读出。

[English](README.md) · 简体中文

</div>

---

## 这是什么

在一个小型开源 LM 上复现 *Jev* 的决策接口（TypeSafe AI 的 "System One"），并检验
一个具体假设：

> **自一致性蒸馏能否把任意 LM 变成 Jev 式的决策模型，蒸馏出的读出是否优于原始的
> 选项 logit 读出？**

Jev 权重不公开，但接口是公开的：它从不生成文本，而是接收 `state` 加若干类型化的
`question`，返回 *choice*、*score* 或 *noul*（是/否概率）以及可校准的置信度。
本仓库实现的是**"读冻结 LM，再把读出蒸馏进一个小头"**这一族方法，并用配对显著性检验、
校准指标和选择性预测曲线和它自己的免费基线对比。

## 主要结论

1. **主要误差来自标签先验，而不是选项顺序。** 仅加一个**免标签的 batch 先验**校正，
   在 26 路 Banking77 上就值 **+0.166 准确率**（`p_raw` 0.644 → 0.810），无需任何训练。
2. **校正必须做进蒸馏目标里。** 事后套到已训练的头会把它*打崩*（0.762 → 0.268），
   因为头已经把 `q*` 里的偏置学进去了。
3. **之后单次前向就能追平多遍上限**：准确率 0.844 / 平衡准确率 0.846、cov@5% 0.600、
   序一致率 0.955；免费基线是 0.644 / 0.632（+0.200，95% CI `[+0.158, +0.244]`，
   McNemar `p = 3.3e-19`）。
4. **头在准确率之外买到的是选择性预测**：5% 错误预算下可自动决策的比例从 0.220 升到
   0.600 —— 这才是决定一条流水线能否自动化的数字。

## 实验设置

所有选项渲染在**同一个上下文**里，一次前向同时得到两种读出：

| 读出 | 含义 |
|---|---|
| `p_raw` | 冻结 LM 在作答位置的标签 token logit 经 softmax —— 免费基线 |
| `p_head` | 一个作用于**每个选项文本末尾**隐状态的小 MLP |

目标来自**自一致性聚合**（paraphrase × 选项顺序的多视图），再做一步**免标签去偏**
（位置偏置与标签先验）。在留出集上比较三者：

1. `p_raw` —— 1 次前向，不训练
2. `q*` —— V 次前向，不训练（单次前向要压缩的上限）
3. `p_head` —— 1 次前向，训练

蒸馏用**对去偏后软目标 `q*` 的前向 KL**，绝不用硬标签 —— 硬标签恰好丢掉了本项目关心的
置信度信号。

> **必须在选项文本末尾读隐状态，绝不能在选项标签字母处读。** 因果 LM 在标签位置还没读到
> 选项内容，在那里读到的头等于在给看不见的文本打分，会静默失效。这是本项目第一个真 bug；
> LoRA 训练里还有第二个同类的目标顺序 bug。两者都记在
> [`docs/RESULTS.md`](docs/RESULTS.md)。

## 结果

基座 `Qwen/Qwen3.5-2B-Base`。完整表格、配对置信区间与命令见
[`docs/RESULTS.md`](docs/RESULTS.md)、[`docs/REPRODUCE.md`](docs/REPRODUCE.md)。

### 26 路意图分类（Banking77，6 视图，n=500）

| 方法 | acc | bal_acc | cov@5% | 序 argmax 一致率 |
|---|---|---|---|---|
| `p_raw` | 0.644 | 0.632 | 0.220 | 0.556 |
| `p_raw` + batch 先验（仍 1 次前向） | 0.810 | 0.810 | 0.250 | – |
| `q*` | 0.772 | 0.765 | 0.420 | – |
| `q*` + batch 先验 + logmean | 0.850 | 0.852 | 0.570 | – |
| `p_head` | 0.762 | 0.751 | 0.440 | 0.928 |
| `p_head`（去偏目标） | **0.844** | **0.846** | **0.600** | **0.955** |
| `p_head_avg`（去偏目标） | **0.852** | **0.853** | 0.582 | – |
| `p_lora_head`（LoRA + 头） | 0.780 | 0.769 | – | 0.966 |

### 3 路 NLI（WANLI，12 视图，n=500）

准确率提升在噪声内；收益在平衡准确率和顺序不变性上。

| 方法 | acc | bal_acc | 序 argmax 一致率 |
|---|---|---|---|
| `p_raw` | 0.612 | 0.473 | 0.449 |
| `p_raw` + batch 先验 | 0.608 | **0.640** | – |
| `q*` + logmean | **0.634** | 0.579 | – |
| `p_head` | 0.624 | 0.560 | 0.898 |
| `p_head`（去偏目标） | 0.612 | 0.639 | **0.912** |

### 77 路意图分类（Banking77 全量，noul 形式）

选项条件的 yes/no 没有标签上限，且结构上就与顺序无关。这里**模型**很关键：base 模型只能
排序、概率接近均匀；instruct 模型（更小！）能给出可用概率。

| 基座 | `p_raw` | `q*` | `p_head` | `p_head` ECE（标定后） | nll |
|---|---|---|---|---|---|
| Qwen3.5-2B-Base | 0.376 | 0.422 | 0.472 | 0.131 | 3.81 |
| Qwen2.5-1.5B-Instruct | 0.472 | 0.504 | **0.556** | **0.044** | 2.13 |

### 全 K 轮换并不划算

把选项摆成全部 K 个循环位移，理论上可让"可加的槽位偏置"在几何平均下精确抵消；但在 K=26 时
它花了 9 倍算力却略**差**（两边都用同一份 batch 先验）：

| Banking77 测试集（n=500） | acc | cov@5% |
|---|---|---|
| 6 视图（3 个随机排列 × 2 个 paraphrase） | **0.842** | **0.576** |
| 52 视图（全部 26 个循环位移 × 2 个 paraphrase） | 0.830 | 0.490 |

循环位移保留相邻关系，因此依赖相对顺序的偏置不会被平均掉，而随机排列会破坏这种结构。
细节见 [`docs/RESULTS.md`](docs/RESULTS.md)。

## 图表

top-label 可靠性曲线（越贴近对角线越好）、选择性预测的 risk-coverage 曲线，以及主要指标并排
比较：

![Top-label reliability](docs/figures/calibration.png)

![Risk-coverage](docs/figures/risk_coverage.png)

![Accuracy, balanced accuracy and coverage](docs/figures/metrics_bars.png)

用 `./run.sh figures` 重新生成（基于 scikit-learn 的 `CalibrationDisplay`，K > 2 时采用
top-label 归约）。

## 快速开始

单卡即可，约 24 GB 显存足够，不需要调度器。

```bash
git clone https://github.com/our0boros/AnyLM2Jev.git && cd AnyLM2Jev

python -m venv .venv && . .venv/bin/activate     # 或：conda env create -f environment.yml
pip install -r requirements.txt

./run.sh data            # 拉取 WANLI + Banking77 到 data/（CPU，需要网络）
./run.sh banking77       # 26 路意图，去偏头 -> runs/eval/banking77_qwen35.json
./run.sh wanli           # 3 路 NLI
./run.sh banking77_full  # 77 路选项条件形式（noul）
./run.sh figures         # -> docs/figures/
```

所有参数都是普通命令行选项，单个阶段也可以手跑：

```bash
PYTHONPATH=src python scripts/10_induce.py \
  --model Qwen/Qwen3.5-2B-Base --data data/banking77_train.jsonl \
  --out runs/induce/banking77_train --paraphrases 2 --max-orders 3 --max-length 768

PYTHONPATH=src python scripts/20_train_head.py \
  --induce runs/induce/banking77_train --out runs/head/banking77_db.pt \
  --views all --epochs 40 --debias batch --combine logmean

PYTHONPATH=src python scripts/30_eval.py \
  --induce runs/induce/banking77_train --head runs/head/banking77_db.pt \
  --test-induce runs/induce/banking77_test --out runs/eval/banking77_qwen35.json
```

`run.sh` 可覆盖的环境变量：`MODEL`、`DEVICE`（分析阶段用 `cpu` 也可以）、`PYTHON`、
`EPOCHS`、`BATCH_SIZE`。

`Qwen/Qwen3.5-2B-Base` 是 `qwen3_5` 混合结构，需要 `transformers>=5.17`。

测试纯 numpy，不需要 GPU：

```bash
python -m unittest discover -s tests
```

## 目录结构

```
run.sh              单卡驱动：data | wanli | banking77 | banking77_full | figures
src/jev/            可复用包
  schema.py         DecisionItem：state + question + options + gold
  prompts.py        paraphrase × 选项顺序视图；choice 与 noul 模板
  modeling.py       冻结 LM 加载、标签读出、可微读出
  induce.py         choice 形式目标：p_raw、q*、每视图隐状态
  noul.py           选项条件 yes/no 目标（任意基数）
  debias.py         免标签位置边缘化 + batch / content-free 先验
  head.py           每选项打分头
  distill.py        前向 KL 训练目标
  calibrate.py      温度标定 + ECE
  metrics.py        acc / 平衡 acc / ECE / Brier / NLL / coverage@risk
  data.py           WANLI 与合成决策集
scripts/            阶段入口（数据、induce、cf 先验、训练、评测、分析、绘图）
tests/              确定性 numpy 测试
docs/               RESULTS、REPRODUCE、figures/
data/ runs/         生成物（已忽略）
```

## 相关工作与引用

Jev 权重不公开，本仓库只复现它公开的**决策接口**。这里用到的免标签校正方法来自校准方向的
公开文献；而把 `p_raw`（对选项标签 logit 的单次前向读出）确立为值得超越的基线，要归功于
接口的一系列开源复现：

- Zhou et al., *Batch calibration: Rethinking calibration for in-context learning and
  prompt engineering*, ICLR 2024.
- Zhao et al., *Calibrate before use: Improving few-shot performance of language
  models*, ICML 2021.
- Zheng et al., *On prompt-driven safeguarding for large language models*, ICLR 2024.
- TypeSafe AI, [Jev](https://typesafe.ai) —— 本仓库复现的接口。
- 该接口的开源复现，包括
  [SemIf / OpenJev](https://github.com/TheoLeeCJ/SemIf) 与
  [AnyJev](https://github.com/nokia-applied-research/AnyJev)。

感谢以上作者提供的方法与基线。

## 本仓库声称与不声称的东西

* **声称**：在一个冻结的 2B 基座上，免标签先验校正 + 自一致性蒸馏到每选项头，相对免费
  logit 读出有大幅且显著的提升，顺序不变性接近完美，选择性预测能力远好于免费基线或多遍聚合。
* **不声称**：冻结特征上的头能超过这些特征本身所能编码的信息。LoRA 的结果说明冻结头是有
  上限的 —— 这正是保留那个开关的原因。
* **待解决**：二元问题的 noul 措辞、与生产级决策 API 的对照（无访问权限）、把选择性预测
  阈值从固定改成学习得到。

## 许可

MIT —— 见 [`LICENSE`](LICENSE)。
