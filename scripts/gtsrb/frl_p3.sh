for trial in 1
do
    dir='../../save_results/gtsrb/percentage30/frl/cluster5_k1_10_k2_10_gamma0.5'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi 
    
    python ../../main_frl.py --trial=$trial \
    --rounds=300 \
    --num_users=100 \
    --frac=0.2 \
    --local_ep=10 \
    --local_bs=20 \
    --lr=0.01 \
    --momentum=0.5 \
    --model=simple-cnn \
    --dataset=gtsrb \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../../save_results/' \
    --partition='percentage30' \
    --alg='frl' \
    --beta=0.1 \
    --local_view \
    --gamma=0.5 \
    --max_iter=500 \
    --nclusters=5 \
    --noise=0 \
    --gpu=3 \
    --print_freq=10 \
    --k2=10 \
    --embedding_dim=64 \
    --hidden_dim=256 \
    --hn_client_mlp_lr=0.001 \
    --hn_client_em_lr=0.005 \
    --hn_cluster_mlp_lr=0.001 \
    --hn_cluster_em_lr=0.005 \
    --activation='relu' \
    --init_way='xavier' \
    --seed=42 \
    --hn_wd=0.001 \
    --model_wd=0.0001 \
    2>&1 | tee $dir'/'$trial'.txt'
done 
