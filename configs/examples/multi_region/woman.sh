export XDG_CACHE=/apdcephfs/share_1290939/richardxia/PretrainedCache
export TORCH_HOME=/apdcephfs/share_1290939/richardxia/PretrainedCache
export HF_HOME=/apdcephfs/share_1290939/richardxia/PretrainedCache
export TOKENIZERS_PARALLELISM=false

name="svd-example2[fps6_mb127-temp]"
config="configs/inference/config_test.yaml"
ckpt="ckpt/model.ckpt"
image_input="testset/woman"
path_ref="testset/reference/woman.png"
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
--s_h 437 1624 2140 85 328 1921 \
--e_h 752 1968 2399 402 728 2336 \
--s_w 324 339 434 243 565 223 \
--e_w 597 515 648 442 771 412 \
--ps_h 531 1718 2209 169 409 2007 \
--pe_h 681 1900 2331 318 552 2261 \
--ps_w 397 396 589 284 646 292 \
--pe_w 523 433 495 381 697 353 \
--x_bais_all 0 \
--x_bais_all 0 \
--x_bais_all 0 \
--x_bais_all 0 \
--x_bais_all 0 \
--x_bais_all 0 \
--y_bais_all 0 \
--y_bais_all 0 \
--y_bais_all 0 \
--y_bais_all 0 \
--y_bais_all 0 \
--y_bais_all 0