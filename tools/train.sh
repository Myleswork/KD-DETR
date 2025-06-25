exp_name=$1
logdir=logs/$exp_name
mkdir -p $logdir
logname=$logdir/nohup.logs
touch $logname

nohup python -m torch.distributed.launch \
             --nproc_per_node=8 \
             kd_main.py \
             -m dab_detr \
             --find_unused_params \
             --output_dir $logdir \
             --batch_size 8 \
             --lr 2e-4 \
             --lr_backbone 2e-5 \
             --epochs 50 \
             --lr_drop 40  \
             --coco_path ../../datasets/coco \
             --backbone resnet18 \
             --num_workers 16 \
             --aux_refpoints \
             --random_refpoints 300 \
             --experiment_name $exp_name \
             > $logname 2>&1 &
