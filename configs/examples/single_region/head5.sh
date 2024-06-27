export XDG_CACHE=/apdcephfs/share_1290939/richardxia/PretrainedCache
export TORCH_HOME=/apdcephfs/share_1290939/richardxia/PretrainedCache
export HF_HOME=/apdcephfs/share_1290939/richardxia/PretrainedCache
export TOKENIZERS_PARALLELISM=false

name="svd-example2[fps6_mb127-temp]"
config="configs/inference/config_test.yaml"
ckpt="ckpt/model.ckpt"
image_input="testset/head5"
path_ref="testset/reference/head5.png"
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
--s_h 828 \
--e_h 1494 \
--s_w 0 \
--e_w 766 \
--ps_h 1157 \
--pe_h 1356 \
--ps_w 529 \
--pe_w 529 \
--x_bias 0 0 0 0 \
--y_bias 10 20 10 0
