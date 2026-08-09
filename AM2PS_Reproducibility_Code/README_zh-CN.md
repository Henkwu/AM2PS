# AM2PS 复现代码快速说明

本项目根据论文 **Adaptive Multi-Prompt and Multi-Scale Feature Fusion for Pneumonia Diagnosis** 重构，是可运行的参考实现，并非从作者原始实验代码恢复得到。

## 最短复现流程

```bash
conda create -n am2ps python=3.10 -y
conda activate am2ps
pip install -r requirements.txt
pip install -e .
python tests/smoke_test.py
```

准备数据后检查目录：

```bash
python scripts/validate_dataset.py data/ChestXRay2017/chest_xray
python scripts/validate_dataset.py data/ChestXRay-Covid19
```

训练完整模型：

```bash
python train.py --config configs/chestxray2017.yaml --device cuda
python train.py --config configs/chestxray_covid19.yaml --device cuda
```

运行消融实验：

```bash
python experiments/run_ablation.py --config configs/chestxray2017.yaml --output-root outputs/ablation_chestxray2017 --device cuda
python experiments/collect_results.py outputs/ablation_chestxray2017 --output outputs/ablation_chestxray2017.csv
```

完整实验流程也可以直接运行：

```bash
bash scripts/run_reproduction.sh
```

论文未明确给出 batch size、随机种子、输入分辨率、数据增强幅度、Adapter 中间维度和注意力头数。因此这些参数在 YAML 中被明确标记为“implementation choice”。此外，论文 Eq. (10)--(11) 和 Eq. (22) 存在执行层面的歧义，详细说明见 `PAPER_IMPLEMENTATION_NOTES.md`。在这些信息未由作者确认前，不应声称代码能够逐数值严格复现论文结果。
