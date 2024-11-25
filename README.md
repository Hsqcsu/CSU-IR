# spectrum
# The spectral structure retrieval of infrared spectra is realized by using Contrast learning from the  thought of CLIP model
# 20241106 The following pre-training method is used for matching pre-training or direct matching training, and the back-end check whether CLIP can be added for comparative learning
![image](https://github.com/user-attachments/assets/fd1f4929-3b8e-45cf-927b-49eb9d97e1a5)
# By extracting functional groups from smiles, three-mode retrieval can be realized during clip training, which makes the application scenarios of infrared retrieval more extensive; More specifically, the model can be applied using the QAR subtask in the VCR domain
# It can also use generative CLIP pre-training for spectral generation of functional groups or smiles
# Modal fusion method：ir_feature is used as the cls_token of smiles_model to re-input smiles_model, and the fused ir_smiles_feature is obtained, and the ir_feature and ir_smiles_feature are compared and learned
# Modal fusion method：Combine ir_feature and smiles_feature for one-to-one attention, or cross attention
# 20241125：CLIP-event training for accurate retrieval of infrared functional groups
