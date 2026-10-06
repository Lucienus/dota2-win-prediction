# 冻结清单与数据重建

此版本准备公开代码、环境规范、冻结比赛 ID 与划分、各时点资格、原始文件哈希及特征字段哈希。GitHub 和匿名入口尚未建立。原始 JSON、特征数组、拟合先验、预测及权重不在包内。

## 包内清单

- `manifests/historical.json`：49,940 场，训练/验证/测试为 34,958 / 7,491 / 7,491；顺序与原实验一致。
- `manifests/later.json`：1,199 场纳入比赛、1 场排除比赛及原因，以及各预测时点的实际比赛顺序。不重新抽样、不替换缺失比赛。
- `manifests/feature_field_hashes.json`：16 份特征数组的每个字段的类型、形状和内容哈希。用于验证重建值，不提供特征值本身。
- `manifests/input_file_hashes.json`：原输入文件 SHA-256。NPZ 压缩容器字节可能变化，因此主要按解压后的字段核对。
- `analysis_context/later_groups.json`：后期比赛的联赛及月份分组，供原统计入口使用。不包含玩家账号或姓名。
- 环境版本和完整特征提取、训练、评估、制表代码随包提供。静态英雄字典及其第三方许可已保留。

## 1. 检查代码和安装环境

使用 Python 3.12，安装到独立环境。以下命令均从本包根目录执行。

```text
python verify_release.py
python -m pip install -r requirements.txt
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m unittest discover -s tests
```

版本记录来自 Windows / CUDA 12.8。CPU 可使用 PyTorch 官方 CPU 源，但不保证跨环境预测逐位一致。部分测试需要未公开的原特征或计时记录，会明确跳过。

## 2. 下载冻结 ID 对应记录

先用 `--limit 1` 测试。删除该参数才会尝试整份清单。重新运行会使用已有文件，不覆盖；每次追加下载日志。

```text
python code/fetch_frozen_matches.py --manifest manifests/historical.json --output ../raw_historical --limit 1
python code/fetch_frozen_matches.py --manifest manifests/later.json --output ../raw_later --limit 1
```

可选密钥仅从环境变量 `OPENDOTA_API_KEY` 读取，不写入代码、清单或日志。遵守提供方的请求额度；默认每次请求至少间隔 1.2 秒，不代表满足所有套餐的日/月额度。遇到 401、403 或 429 会停止，先处理访问或额度问题再继续。不自动付费、不请求重新解析比赛、不用其他比赛补齐缺口。

哈希按原始文件字节计算。仅空白、换行或 JSON 键顺序变化也可能造成不同。当前没有为全部原始 JSON 提供规范化内容哈希。日志中的 `byte_hash_differs` 不能独自证明特征值发生变化，也不能证明值未变化。

## 3. 重建历史特征和训练先验

默认要求原始文件哈希全部一致。输出必须是新目录，缺失文件或提取错误会停止，不自动删除样本。失败目录不能用于论文实验。

```text
python code/rebuild_frozen.py --cohort historical --raw-dir ../raw_historical --output inputs/historical
```

先验只从冻结训练划分拟合。使用重新获取、字节哈希不同的数据时，必须显式加 `--allow-changed-raw`，并使用另一个输出目录。该选项不改原清单、不跳过缺失比赛；重建报告记录差异。所有特征字段均匹配时，可确认当前提取输出与冻结字段一致；不匹配时属于变化数据上的重做实验，不能写成精确复现，也不预先承诺“等效数据集”。

## 4. 重建后期评估特征

```text
python code/rebuild_frozen.py --cohort later --raw-dir ../raw_later --prior inputs/historical/purchase_prior.json --output inputs/later
```

使用历史训练先验，绝不从后期队列拟合先验。按冻结的各时点 ID 顺序提取；评估专用辅助标签和掩码保持原文件的零值。每次输出 `rebuild_status.json`，必须查看 `features_equal_frozen` 和差异清单。不要混用严格复现输入与变化数据输入。

## 5. 训练、评估及制表

```text
python run_all.py --output ../new_runs
python code/evaluate_later.py --runs ../new_runs --later inputs/later --output ../new_later_results --scope paper
python code/aggregate_results.py --runs ../new_runs --later-results ../new_later_results --output ../new_tables
```

完整网格是 168 个历史选定配置及 84 个后期评估。这里提供执行方法，不表示该版本已重新训练全部配置。训练和比较结果须与原论文记录分开保存。

## 仍有的限制

哈希不能恢复丢失的数据。若 OpenDota 不再提供某场比赛、关键字段变化，单靠这份清单可能无法重建论文数据。冻结原始记录仍由作者本地保存；可联系作者申请，实际提供需评估适用条款。审稿期间拟经编辑部联系作者，发表后使用通讯作者联系方式。

本材料提高了样本与流程的可核查性，但不等于已保证长期数据可用，也不等于期刊已认可其符合全部复现要求。公开前仍须建立匿名访问入口、记录 GitHub 版本和 commit，并检查实名版权信息在审稿副本中的处理。

官方参考：https://docs.opendota.com/ ；https://transactions.games/submit/submission-guidelines
