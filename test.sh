CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29378 \
    main_semi.py  \
    --output_dir ./experiments/test \
    --ckpt_path ./pretrained_models/burn_in.pth \
    --num_stages 3 \
    --num_points 200 \
    --num_workers 0 \
    --batch_size 12 \
    --num_gpu 1 \
    --test_only  \
    $@
    

