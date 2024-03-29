python train.py --data_dir=data \
                --ckpt_path=ckpts/xnet_kin_90.pth.tar \
                --save_dir=logs \
		--name=Test \
		--abnormal_prob=0.3 \
                --agg_method=max \
                --batch_size=8 \
                --best_ckpt_metric=val_AUROC \
                --crop_shape=192,192 \
                --cudnn_benchmark=False \
                --dataset=pe \
                --do_classify=True \
                --epochs_per_eval=1 \
                --epochs_per_save=1 \
                --fine_tune=True \
                --fine_tuning_boundary=classifier \
                --fine_tuning_lr=1e-2 \
                --gpu_ids=-1 \
                --include_normals=True \
                --iters_per_print=8 \
                --iters_per_visual=8000 \
                --learning_rate=1e-2 \
                --lr_decay_step=600000 \
                --lr_scheduler=cosine_warmup \
                --lr_warmup_steps=10000 \
                --model=PENetClassifier\
                --model_depth=50 \
                --num_classes=1 \
                --num_epochs=1 \
                --num_slices=24 \
                --num_visuals=8 \
                --num_workers=8 \
                --optimizer=sgd \
                --pe_types='["central", "segmental"]' \
                --resize_shape=208,208 \
                --sgd_dampening=0.9 \
                --sgd_momentum=0.9 \
                --use_hem=False \
                --use_pretrained=True \
                --weight_decay=1e-3
