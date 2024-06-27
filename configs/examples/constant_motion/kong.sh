name="svd-example2[fps6_mb127-temp]"
config="configs/inference/config_test.yaml"
ckpt="ckpt/model.ckpt"
image_input="testset/kong"
path_ref="testset/reference/kong.png"
res_dir="outputs"

python3 main/inference/sample_constant_motion.py \
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
--s_h 290 \
--e_h 1128 \
--s_w 0 \
--e_w 456 \
--ps_h 720 911 835 877 \
--ps_w 223 258 150 123
