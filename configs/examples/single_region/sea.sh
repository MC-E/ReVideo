name="svd-example2[fps6_mb127-temp]"
config="configs/inference/config_test.yaml"
ckpt="ckpt/model.ckpt"
image_input="testset/sea2"
path_ref="testset/reference/sea2.png"
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
--s_h 547 \
--e_h 1400 \
--s_w 488 \
--e_w 882 \
--ps_h 939 \
--pe_h 1026 \
--ps_w 649 \
--pe_w 687 \
--x_bias 0 0 0 0 \
--y_bias 10 20 10 0
