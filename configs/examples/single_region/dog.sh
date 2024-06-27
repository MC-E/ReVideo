name="svd-example2[fps6_mb127-temp]"
config="configs/inference/config_test.yaml"
ckpt="ckpt/model.ckpt"
image_input="testset/dog"
path_ref="testset/reference/dog.png"
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
--s_h 83 \
--e_h 1477 \
--s_w 97 \
--e_w 982 \
--ps_h 803 608 1022 \
--pe_h 574 529 811 \
--ps_w 429 261 340 \
--pe_w 429 261 340 \
--x_bias 0 \
--y_bias 0
