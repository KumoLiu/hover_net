python run_infer.py \
--gpu='0, 1' \
--nr_types=5 \
--type_info_path=type_info.json \
--batch_size=64 \
--model_mode=fast \
--model_path=/home/yunliu/Workspace/Code/tutorials/pathology/hovernet/model.pt \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
tile \
--input_dir=/home/yunliu/Workspace/Data/CoNSeP/Test/Images/ \
--output_dir=/home/yunliu/Workspace/Data/CoNSeP/Test-monai/nv-imagenet-weight-v2/fast321 \
--mem_usage=0.1 \
--draw_dot \
--save_raw_map \