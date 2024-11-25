source /home/bpt3/miniconda3/etc/profile.d/conda.sh
conda activate bpt-env
# navigating to the directory where the code is present
cd /home/bpt3/code/UniViT
python -m src.baselines.segmentation.test_iBOT_newMetric
clear
python -m src.baselines.segmentation.test_iBOT_newMetric
exit
