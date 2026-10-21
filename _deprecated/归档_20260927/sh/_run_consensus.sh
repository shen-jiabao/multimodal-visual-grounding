set -e
DS='/root/dataset/初赛数据集-基于大模型的多模态视觉理解与推理'
ln -sf /root/infer/result_gd_3modal_v2.json /root/output/result_gd_3modal_v2.json
ln -sf "$DS/queries/queries.json" /root/infer/queries.json
ls -l /root/output/result_gd_3modal_v2.json /root/infer/queries.json
cd /root/infer && /root/miniconda3/envs/myconda/bin/python -u _fusion_consensus.py 2>&1 | tail -60
