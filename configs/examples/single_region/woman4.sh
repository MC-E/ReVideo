name="svd-example2[fps6_mb127-temp]"
config="configs/inference/config_test.yaml"
ckpt="ckpt/model.ckpt"
image_input="testset/woman4"
path_ref="testset/reference/woman4.png"
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
--s_h 94 \
--e_h 613 \
--s_w 236 \
--e_w 952 \
--ps_h 278 \
--pe_h 420 \
--ps_w 640 \
--pe_w 640 \
--x_bias 0 \
--y_bias 0
