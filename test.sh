






###---------------->
# model='/home/jinlongli/personal/personal_jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/CoBEVT'

# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/0.proposed_comamba_multi-scale_4+4_mean_max_2024_05_01_13_04_54'


# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/2.comamba_normal_multi-scale_only_2024_05_03_23_56_39'

# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/2.5x_comamba_normal_multi-scale_vssfusion_2024_05_04_01_23_36'

# model='/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/v2x-vit'

# model='/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/attfuse'




#------------------------>

# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/3.opv2v_mamba_8448_3_2024_05_05_00_38_14'



# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/opv2v_2024_05_04_22_54_34'

# model='/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/CoBEVT'


#--------------------------------------------->





model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/8.V2X_opv2v_mamba_max_mean_2024_05_25_02_52_06'




# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/6.opv2v_mamba_max_mean_opv2v_2024_05_15_11_00_45'






# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/6.Cobevt_V2XSet_2024_05_20_19_45_00'


# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/8.V2X_V2VAM_2024_05_26_18_29_42'

# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/8.V2X_V2VAM_2024_05_26_18_29_42'
# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/6.opv2v_where2comm_2024_05_31_14_36_44'


# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/9.aba_v2x_baseline_only_2024_05_31_14_41_06'
# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/9.aba_v2x_amx_mean_only_2024_05_31_14_43_12'
# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/9.aba_v2x_mamba_only_2024_05_31_14_44_16'


#conda activate comamba



#conda activate comamba
# run python script
CUDA_VISIBLE_DEVICES=0 python3 opencood/tools/inference.py \
    --fusion_method intermediate \
    --model_dir $model 
    # --show_vis