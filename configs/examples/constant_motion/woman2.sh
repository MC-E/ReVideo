export XDG_CACHE=/apdcephfs/share_1290939/richardxia/PretrainedCache
export TORCH_HOME=/apdcephfs/share_1290939/richardxia/PretrainedCache
export HF_HOME=/apdcephfs/share_1290939/richardxia/PretrainedCache
export TOKENIZERS_PARALLELISM=false

name="svd-example2[fps6_mb127-temp]"
config="configs/inference/config_test.yaml"
ckpt="ckpt/model.ckpt"
image_input="testset/woman2"
path_ref="testset/reference/woman2.png"
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
--s_h 797 \
--e_h 1098 \
--s_w 315 \
--e_w 647 \
--ps_h 888 1009 950 \
--ps_w 391 385 502
