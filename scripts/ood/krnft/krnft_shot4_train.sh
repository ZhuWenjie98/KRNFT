# ############################################################  Final, zero-shot performance.
tau=1.0
beta=1.0
######################### clip + negood +  Neglabel, OOD.
in_score=sum
prompt=simple
#  30 50 100 200 1000 2000
# guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
    #    --evaluator.name ood_clip \
#for prompt in vanilla good bad small large simple nice
#--dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
for prompt in nice
do
    for beta in 1
    do
        for batch_size in 1
        do
            for lr in 1e-5
            do 
                for shot in 4
                do
                    for seed in 0
                    do               
                        CUDA_VISIBLE_DEVICES=0 python main.py \
                        --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
                        configs/networks/krnft.yml \
                        configs/pipelines/train/train_krnft.yml \
                        configs/preprocessors/randcrop_preprocessor.yml \
                        configs/postprocessors/mcm.yml \
                        --dataset.train.batch_size ${batch_size} \
                        --dataset.val.batch_size ${batch_size} \
                        --dataset.test.batch_size ${batch_size} \
                        --ood_dataset.batch_size ${batch_size} \
                        --dataset.train.few_shot ${shot} \
                        --dataset.num_classes 11500 \
                        --evaluator.name ood_clip \
                        --network.name clip_krnft \
                        --network.backbone.text_prompt ${prompt} \
                        --network.backbone.OOD_NUM 10500 \
                        --network.backbone.meta_dim 64 \
                        --network.pretrained False \
                        --seed ${seed} \
                        --trainer.name krnft \
                        --postprocessor.name oneoodpromptdevelop \
                        --postprocessor.postprocessor_args.tau ${tau}  \
                        --postprocessor.postprocessor_args.beta ${beta}  \
                        --postprocessor.postprocessor_args.in_score ${in_score}  \
                        --optimizer.lr ${lr}  \
                        --num_gpus 1 --num_workers 6 \
                        --merge_option merge \
                        --output_dir ./results/krnft/meta_dim/dim8/ \
                        --mark shot${shot}_batch_size${batch_size}_lr${lr}
                    done    
                done
            done
        done    
    done
done
   