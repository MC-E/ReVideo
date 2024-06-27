name="svd-example2[fps6_mb127-temp]"
config="configs/inference/config_test.yaml"
ckpt="ckpt/model.ckpt"
image_input="testset/woman5"
path_ref="testset/reference/woman5.png"
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
--s_h 605 \
--e_h 1449 \
--s_w 599 \
--e_w 1929 \
--ps_h 825 1263 1671 \
--ps_w 1123 1287 1231
