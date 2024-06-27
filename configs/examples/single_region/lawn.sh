name="svd-example2[fps6_mb127-temp]"
config="configs/inference/config_test.yaml"
ckpt="ckpt/model.ckpt"
image_input="testset/lawn"
path_ref="testset/reference/lawn.png"
res_dir="outputs"

python3 main/inference/sample_single_region.py \
--seed 23 \
--ckpt $ckpt \
--config $config \
--savedir $res_dir/$name \
--savefps 10 \
--ddim_steps 15 \
--frames 14 \
--savefps 10 \
--input $image_input \
--path_ref $path_ref \
--fps 10 \
--motion 127 \
--cond_aug 0.02 \
--decoding_t 1 --resize \
--s_h 383 \
--e_h 1109 \
--s_w 485 \
--e_w 1627 \
--ps_h 717 741 \
--pe_h 717 741 \
--ps_w 731 945 \
--pe_w 781 995 \
--x_bias 15 30 45 30 15 0 \
--y_bias 0
