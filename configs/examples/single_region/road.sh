name="svd-example2[fps6_mb127-temp]"
config="configs/inference/config_test.yaml"
ckpt="ckpt/model.ckpt"
image_input="testset/road"
path_ref="testset/reference/road.png"
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
--s_h 886 \
--e_h 1235 \
--s_w 704 \
--e_w 944 \
--ps_h 1115 \
--pe_h 1046 \
--ps_w 790 \
--pe_w 881 \
--x_bias 0 \
--y_bias 0
