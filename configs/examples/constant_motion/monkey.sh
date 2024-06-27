name="svd-example2[fps6_mb127-temp]"
config="configs/inference/config_test.yaml"
ckpt="ckpt/model.ckpt"
image_input="testset/monkey"
path_ref="testset/reference/monkey.png"
res_dir="outputs"

python3 main/inference/sample_constant_motion.py \
--seed 23 \
--ckpt $ckpt \
--config $config \
--savedir $res_dir/$name \
--savefps 10 \
--ddim_steps 25 \
--frames 14 \
--savefps 10 \
--input $image_input \
--path_ref $path_ref \
--fps 10 \
--motion 127 \
--cond_aug 0.02 \
--decoding_t 1 --resize \
--s_h 461 \
--e_h 1039 \
--s_w 189 \
--e_w 770 \
--ps_h 615 736 694 \
--ps_w 457 461 586
