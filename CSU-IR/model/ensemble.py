import torch

# Replace the path with your own trained model weights
checkpoints = [
    (r'D:\Spectrum\1122_after\model\ESA_model_sigmoid\20250530_esa_ir_CNN_transformer\trained\8.50.51\79w_pretrain_detanet_finetuned_augreal_finetuned_650-4000\best_smiles_model_epoch_71_ratio_0.9079.pth',0.9079),
    (r'D:\Spectrum\1122_after\model\ESA_model_sigmoid\20250530_esa_ir_CNN_transformer\trained\8.50.51\79w_pretrain_detanet_finetuned_augreal_finetuned_650-4000\best_smiles_model_epoch_77_ratio_0.9089.pth',0.9089),
    (r'D:\Spectrum\1122_after\model\ESA_model_sigmoid\20250530_esa_ir_CNN_transformer\trained\8.50.51\79w_pretrain_detanet_finetuned_augreal_finetuned_650-4000\best_smiles_model_epoch_78_ratio_0.9084.pth',0.9084),
    (r'D:\Spectrum\1122_after\model\ESA_model_sigmoid\20250530_esa_ir_CNN_transformer\trained\8.50.51\79w_pretrain_detanet_finetuned_augreal_finetuned_650-4000\best_smiles_model_epoch_79_ratio_0.9094.pth',0.9094),
    (r'D:\Spectrum\1122_after\model\ESA_model_sigmoid\20250530_esa_ir_CNN_transformer\trained\8.50.51\79w_pretrain_detanet_finetuned_augreal_finetuned_650-4000\best_smiles_model_epoch_80_ratio_0.9089.pth',0.9089),
]


models = [torch.load(checkpoint[0]) for checkpoint in checkpoints]
top1_ratios = [checkpoint[1] for checkpoint in checkpoints]

total_weight = sum(top1_ratios)


average_model = models[0].copy()
for key in average_model.keys():
    average_model[key] = average_model[key] * top1_ratios[0]

for i in range(1, len(models)):
    for key in average_model.keys():
        average_model[key] += models[i][key] * top1_ratios[i]

for key in average_model.keys():
    average_model[key] /= total_weight

torch.save(average_model,r'D:\Spectrum\1122_after\model\ESA_model_sigmoid\20250530_esa_ir_CNN_transformer\trained\8.50.51\79w_pretrain_detanet_finetuned_augreal_finetuned_650-4000\best_smiles_model_650-4000.pth')