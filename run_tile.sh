python run_infer.py \
--gpu='0' \
--nr_types=5 \
--type_info_path=type_info.json \
--batch_size=64 \
--model_mode=original \
--model_path=/home/yunliu/Workspace/Code/tutorials/pathology/hovernet/runs/Nov12_14-20-19_yunliu-MS-7D31bs8_ep50_lr0.0001_5_raw_1_hue-hover-affine/model_49.pth \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
tile \
--input_dir=/home/yunliu/Workspace/Data/CoNSeP/Test/Images/ \
--output_dir=/home/yunliu/Workspace/Data/CoNSeP/TestNov12_14-20-19/ \
--mem_usage=0.1 \
--draw_dot \
--save_raw_map \
--save_qupath
