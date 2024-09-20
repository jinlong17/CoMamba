
#conda activate comamba




# model='/home/jinlongli/1.Detection_Set/TIV_LCRN/NEW_log/4.lcrn/old/point_pillar_intermediate_V2VAM_nocompression_2024_01_06_01_10_11/net_epoch19.pth'

# hypes_yaml='/home/jinlongli/1.Detection_Set/Comamba/V2V/opencood/hypes_yaml/point_pillar_intermediate_V2VAM.yaml'




# hypes_yaml='/home/jinlongli/1.Detection_Set/Comamba/V2V/opencood/hypes_yaml/point_pillar_comamba.yaml'


# hypes_yaml='/home/jinlongli/1.Detection_Set/Comamba/V2V/opencood/hypes_yaml/point_pillar_comamba_V1.yaml'
# hypes_yaml='/home/jinlongli/1.Detection_Set/Comamba/V2V/opencood/hypes_yaml/point_pillar_comamba_V2.yaml'


# hypes_yaml='/home/jinlongli/1.Detection_Set/Comamba/V2V/opencood/hypes_yaml/point_pillar_opv2v.yaml'
hypes_yaml='/home/jinlongli/1.Detection_Set/Comamba/V2V/opencood/hypes_yaml/point_pillar_opv2v_comamba.yaml'




# hypes_yaml='/home/jinlongli/1.Detection_Set/Comamba/V2V/opencood/hypes_yaml/point_pillar_cobevt.yaml'
# hypes_yaml='/home/jinlongli/1.Detection_Set/Comamba/V2V/opencood/hypes_yaml/point_pillar_v2xvit.yaml'



# hypes_yaml='/home/jinlongli/1.Detection_Set/Comamba/V2V/opencood/hypes_yaml/point_pillar_intermediate_V2VAM.yaml'

# hypes_yaml='/home/jinlongli/1.Detection_Set/Comamba/V2V/opencood/hypes_yaml/point_pillar_where2comm.yaml'


# run python script
########-------------------------------------------------------------------

# CUDA_VISIBLE_DEVICES=4 python3 opencood/tools/train_real.py  --hypes_yaml $hypes_yaml  #--model $model  #--model_dir $path #--model_dir $path #--model $model 


CUDA_VISIBLE_DEVICES=5 python3 opencood/tools/train.py  --hypes_yaml $hypes_yaml  #--model $model  #--model_dir $path #--model_dir $path #--model $model 





# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node=5  --use_env opencood/tools/train.py --hypes_yaml $hypes_yaml #[--model_dir  ${CHECKPOINT_FOLDER}]


