python main_two_res_3d_recon_with_base.py --dataset celebA --batch_size 1  --sample_size 1 --learning_rate 0.001 --image_size 224 \
  --gf_dim 32 --df_dim 32 --dfc_dim 320 --gfc_dim 640 --z_dim 20 --c_dim 3 --is_partbase_albedo True --is_reduce True --gpu 0\
  --checkpoint_dir ../checkpoints/
