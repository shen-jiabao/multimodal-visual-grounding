# 差异报告：pick_v1.zip vs 27b_annotate.zip

- 基体：`result_semi_27b_annotate.zip`（对方管线 10-05 复跑全量）
- 本件：`result_semi_pick_v1.zip` = 基体 + 人眼选蓝位贴 `result_semi_q38_nonord`（0.7963 件）框
- 条目数：两件均 5690；键集与四字段回显完全一致

## 一、总览

| 指标 | 值 |
|---|---|
| 差异条目（bbox 不同） | **54 / 5690**（0.9%） |
| 其余条目 | 逐位相同 |
| 差异中序数类查询 | 5 |
| 差异中非序数 | 49 |
| 有效分歧位（IoU<0.5，可能翻分） | 11 |
| 无效改动（IoU>=0.5，两件同侧） | 43 |

## 二、按 IoU(annotate, q38) 分层

| 层 | 条数 |
|---|---|
| 0（完全异区域） | 8 |
| 0<iou<0.3 | 3 |
| 0.3<=iou<0.5 | 0 |
| iou>=0.5（无效改动） | 43 |

## 三、逐条清单（按 qid）

| # | qid | 序数 | IoU | 查询 | annotate 框 | pick_v1 框(=q38) | 面积变化 |
|---|---|---|---|---|---|---|---|
| 1 | 000015_005 |  | 0.52 | The dark floor tile closest to the camera | [0.537, 0.944, 0.62, 1.0] | [0.538, 0.902, 0.61, 0.999] | ×1.50 |
| 2 | 000022_002 |  | 0.65 | A person wearing a white shirt sitting at an out | [0.886, 0.577, 0.936, 0.69] | [0.888, 0.578, 0.936, 0.655] | ×0.65 |
| 3 | 000064_002 |  | 0.72 | Three poles standing on the left side of the lak | [0.457, 0.249, 0.488, 0.332] | [0.453, 0.248, 0.494, 0.335] | ×1.39 |
| 4 | 000113_004 | 序 | 0.00 | From left to right, the first red letter V | [0.357, 0.539, 0.403, 0.647] | [0.105, 0.57, 0.135, 0.7] | ×0.79 |
| 5 | 000113_005 | 序 | 0.00 | From left to right, the first red letter E | [0.454, 0.523, 0.522, 0.709] | [0.15, 0.562, 0.192, 0.692] | ×0.43 |
| 6 | 000141_003 |  | 0.95 | Large gray commercial building with 'MIXC ON! WO | [0.701, 0.301, 1.0, 0.875] | [0.712, 0.302, 1.0, 0.881] | ×0.97 |
| 7 | 000141_004 |  | 0.77 | White car parked on the left side of the street | [0.0, 0.696, 0.083, 0.815] | [0.0, 0.697, 0.065, 0.816] | ×0.78 |
| 8 | 000211_002 |  | 0.60 | The large illuminated letter 'U' with embedded w | [0.519, 0.372, 0.741, 0.79] | [0.518, 0.418, 0.671, 0.792] | ×0.62 |
| 9 | 000267_002 |  | 0.00 | A woman wearing a white top and hat in the dista | [0.27, 0.417, 0.283, 0.466] | [0.299, 0.417, 0.316, 0.466] | ×1.31 |
| 10 | 000281_006 |  | 0.28 | The blue and white bumper car with a person in a | [0.125, 0.369, 0.282, 0.484] | [0.208, 0.365, 0.284, 0.438] | ×0.31 |
| 11 | 000281_007 |  | 0.93 | The green and white bumper car parked behind the | [0.295, 0.344, 0.388, 0.42] | [0.295, 0.344, 0.387, 0.425] | ×1.05 |
| 12 | 000281_008 |  | 0.97 | The person in a blue patterned shirt standing at | [0.901, 0.199, 1.0, 0.821] | [0.9, 0.207, 1.0, 0.823] | ×1.00 |
| 13 | 000281_009 |  | 0.96 | The child in a white and green shirt standing be | [0.889, 0.407, 0.958, 0.75] | [0.888, 0.403, 0.959, 0.75] | ×1.04 |
| 14 | 000281_010 |  | 0.82 | The person in a striped shirt standing at the ri | [0.883, 0.226, 0.92, 0.447] | [0.881, 0.224, 0.925, 0.452] | ×1.23 |
| 15 | 000281_011 |  | 0.84 | The red sign with white text mounted above the r | [0.944, 0.077, 1.0, 0.163] | [0.936, 0.073, 1.0, 0.163] | ×1.20 |
| 16 | 000281_012 |  | 0.78 | The brown sign with black text at the left | [0.005, 0.342, 0.035, 0.375] | [0.006, 0.345, 0.037, 0.373] | ×0.88 |
| 17 | 000281_013 |  | 0.87 | A person wearing a mask | [0.303, 0.362, 0.353, 0.469] | [0.305, 0.362, 0.35, 0.465] | ×0.87 |
| 18 | 000288_001 |  | 0.96 | A red and white bumper car with a man in a black | [0.368, 0.41, 0.682, 0.695] | [0.37, 0.41, 0.675, 0.693] | ×0.96 |
| 19 | 000288_002 |  | 0.96 | A blue and white bumper car with a boy in a gree | [0.844, 0.332, 0.993, 0.547] | [0.846, 0.333, 0.994, 0.55] | ×1.00 |
| 20 | 000288_003 |  | 0.96 | A green and white bumper car with no driver park | [0.181, 0.29, 0.314, 0.396] | [0.18, 0.288, 0.314, 0.394] | ×1.01 |
| 21 | 000288_004 |  | 0.92 | A blue and white bumper car with a man in a whit | [0.574, 0.262, 0.669, 0.355] | [0.574, 0.257, 0.67, 0.353] | ×1.04 |
| 22 | 000288_005 |  | 0.86 | A red and white bumper car with no driver parked | [0.67, 0.267, 0.749, 0.372] | [0.666, 0.265, 0.75, 0.365] | ×1.01 |
| 23 | 000288_006 |  | 0.88 | A green and white bumper car with no driver park | [0.944, 0.262, 1.0, 0.355] | [0.938, 0.262, 1.0, 0.352] | ×1.07 |
| 24 | 000288_007 |  | 0.95 | A man in a black shirt driving a red and white b | [0.438, 0.317, 0.546, 0.568] | [0.438, 0.315, 0.548, 0.562] | ×1.00 |
| 25 | 000288_009 |  | 0.96 | A man in a white shirt driving a blue and white  | [0.606, 0.229, 0.65, 0.283] | [0.606, 0.229, 0.65, 0.285] | ×1.04 |
| 26 | 000288_010 |  | 0.54 | A red taillight on the green and white bumper ca | [0.276, 0.327, 0.296, 0.343] | [0.277, 0.32, 0.299, 0.345] | ×1.72 |
| 27 | 000288_011 |  | 0.62 | A red taillight on the blue and white bumper car | [0.63, 0.296, 0.648, 0.312] | [0.627, 0.293, 0.646, 0.313] | ×1.32 |
| 28 | 000288_012 |  | 0.95 | A green ball in the bottom left corner of the im | [0.0, 0.837, 0.026, 1.0] | [0.0, 0.835, 0.027, 1.0] | ×1.05 |
| 29 | 000302_001 |  | 0.56 | The white Chinese text on the green sign | [0.575, 0.593, 0.598, 0.607] | [0.573, 0.593, 0.602, 0.613] | ×1.80 |
| 30 | 000302_002 |  | 0.69 | The small animal illustrations at the bottom of  | [0.565, 0.689, 0.607, 0.739] | [0.564, 0.702, 0.607, 0.737] | ×0.72 |
| 31 | 000393_003 | 序 | 0.08 | The deer with antlers that is second closest to  | [0.406, 0.177, 0.471, 0.37] | [0.444, 0.261, 0.543, 0.523] | ×2.07 |
| 32 | 000414_002 |  | 0.00 | The person wearing white clothes on the leftmost | [0.121, 0.408, 0.136, 0.449] | [0.071, 0.437, 0.085, 0.492] | ×1.25 |
| 33 | 000414_006 |  | 0.95 | The bear striding forward to the back | [0.362, 0.545, 0.451, 0.652] | [0.36, 0.544, 0.452, 0.653] | ×1.05 |
| 34 | 000414_007 |  | 0.71 | The dark-colored umbrella | [0.235, 0.417, 0.251, 0.441] | [0.233, 0.415, 0.253, 0.442] | ×1.41 |
| 35 | 000414_008 |  | 0.96 | The bear looking down for something | [0.93, 0.637, 1.0, 0.833] | [0.928, 0.635, 1.0, 0.833] | ×1.04 |
| 36 | 000421_001 |  | 0.98 | The bear looking at the camera | [0.399, 0.382, 0.548, 0.788] | [0.399, 0.38, 0.55, 0.788] | ×1.02 |
| 37 | 000421_002 |  | 0.92 | The stone being stepped on by the bear | [0.402, 0.775, 0.538, 0.961] | [0.4, 0.772, 0.54, 0.968] | ×1.08 |
| 38 | 000421_003 |  | 1.00 | The longest iron pipe | [0.0, 0.607, 0.526, 1.0] | [0.0, 0.607, 0.527, 1.0] | ×1.00 |
| 39 | 000421_004 |  | 0.98 | The columnar object oriented nearly perpendicula | [0.081, 0.922, 0.116, 0.981] | [0.081, 0.921, 0.116, 0.981] | ×1.02 |
| 40 | 000428_002 |  | 0.92 | The wooden pillar | [0.612, 0.595, 0.852, 1.0] | [0.617, 0.57, 0.852, 0.997] | ×1.03 |
| 41 | 000428_003 |  | 0.99 | The brown bear in the left | [0.281, 0.31, 0.584, 0.681] | [0.28, 0.307, 0.584, 0.68] | ×1.01 |
| 42 | 000428_004 |  | 0.99 | The iron pipe relatively on the far left | [0.28, 0.563, 0.63, 1.0] | [0.278, 0.563, 0.63, 1.0] | ×1.01 |
| 43 | 000428_005 |  | 0.97 | The iron pillar on the right slanting upward lef | [0.693, 0.561, 0.895, 0.831] | [0.693, 0.563, 0.895, 0.826] | ×0.97 |
| 44 | 000428_006 |  | 0.99 | The net near the camera | [0.281, 0.563, 0.852, 1.0] | [0.278, 0.562, 0.848, 1.0] | ×1.00 |
| 45 | 000428_007 |  | 0.88 | The largest stone at the top left | [0.091, 0.025, 0.134, 0.072] | [0.089, 0.025, 0.137, 0.071] | ×1.09 |
| 46 | 000442_001 |  | 0.81 | The white crane closest to the camera | [0.572, 0.568, 0.624, 0.897] | [0.579, 0.567, 0.625, 0.882] | ×0.85 |
| 47 | 000442_004 |  | 0.84 | The white crane on the far left stepping in the  | [0.501, 0.188, 0.546, 0.268] | [0.501, 0.188, 0.542, 0.262] | ×0.84 |
| 48 | 000442_006 |  | 0.96 | The white crane standing the straightest | [0.733, 0.498, 0.791, 0.762] | [0.733, 0.496, 0.792, 0.757] | ×1.01 |
| 49 | 000442_007 |  | 0.99 | The stone on the far left | [0.0, 0.774, 0.198, 1.0] | [0.0, 0.773, 0.2, 1.0] | ×1.01 |
| 50 | 000449_004 |  | 0.00 | The white crane on the far left looking to the l | [0.137, 0.071, 0.182, 0.117] | [0.271, 0.111, 0.329, 0.188] | ×2.16 |
| 51 | 000449_005 |  | 0.00 | The white crane in the middle walking backward w | [0.459, 0.117, 0.49, 0.248] | [0.504, 0.472, 0.547, 0.643] | ×1.81 |
| 52 | 000456_002 |  | 0.23 | The green bottom of the green railing on the far | [0.0, 0.447, 0.352, 1.0] | [0.0, 0.706, 0.152, 0.999] | ×0.23 |
| 53 | 000680_003 | 序 | 0.00 | The first electric vehicle from bottom to top | [0.233, 0.307, 0.568, 0.628] | [0.854, 0.421, 1.0, 0.998] | ×0.78 |
| 54 | 000680_004 | 序 | 0.00 | The trunk of the first electric vehicle from top | [0.579, 0.215, 0.765, 0.42] | [0.854, 0.422, 1.0, 0.997] | ×2.20 |

## 四、期望影响

每处有效分歧位（IoU<0.5）：人眼判对 +1 条、判错 −1 条；IoU≥0.5 的 43 处两件几乎同侧，ACC@0.5 下期望贡献 ≈ 0。
净期望 = 11 × (2p−1) 条（p=判对率）。p=0.7 ⇒ +8.8 条 ⇒ +0.0015。
