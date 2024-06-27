name="svd-example2[fps6_mb127-temp]"
config="configs/inference/config_test.yaml"
ckpt="ckpt/model.ckpt"
image_input="testset/lawn2"
path_ref="testset/reference/lawn2.png"
res_dir="outputs"

python3 main/inference/sample_multi_region.py \
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
--fps 6 \
--motion 127 \
--cond_aug 0.02 \
--decoding_t 1 --resize \
--s_h 1579 2577 509 1405 2369 \
--e_h 2155 2991 821 1723 2725 \
--s_w 311 119 1207 1257 1189 \
--e_w 673 393 1653 1663 1641 \
--ps_h 2887 2027 675 1577 2561 \
--pe_h 2707 1777 675 1577 2561 \
--ps_w 221 449 1347 1395 1367 \
--pe_w 221 449 1347 1395 1367 \
--x_bias_all 0 \
--x_bias_all 0 \
--x_bias_all 10 20 30 20 10 0 \
--x_bias_all 10 20 30 20 10 0 \
--x_bias_all 10 20 30 20 10 0 \
--y_bias_all 0 \
--y_bias_all 0 \
--y_bias_all 0 \
--y_bias_all 0 \
--y_bias_all 0 