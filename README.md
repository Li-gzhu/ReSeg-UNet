# ReSeg-UNet  
We will continue to update.

🔧 Core Logic: Reconstruction-Guided Feature Consistency Loss
This code implements a reconstruction-guided feature consistency loss to enhance the segmentation performance of medical image models. The idea is to enforce consistency between the encoder and decoder features of the segmentation branch and the reconstruction branch.

🧩 Feature Extraction
We extract multi-level features from both branches:

# Segmentation encoder features
se_encoder_feature1 = outputs[2][2]
se_encoder_feature2 = outputs[2][1]
se_encoder_feature3 = outputs[2][0]

# Reconstruction encoder features
re_encoder_feature1 = outputs2[2][2]
re_encoder_feature2 = outputs2[2][1]
re_encoder_feature3 = outputs2[2][0]

# Segmentation decoder features
se_decoder_feature1 = outputs[3][0]
se_decoder_feature2 = outputs[3][1]
se_decoder_feature3 = outputs[3][2]

# Reconstruction decoder features
re_decoder_feature1 = outputs2[3][0]
re_decoder_feature2 = outputs2[3][1]
re_decoder_feature3 = outputs2[3][2]

📉 Feature Consistency Loss
To enforce consistency, we compute the mean squared error (MSE) between corresponding encoder and decoder features from the two branches:

# Encoder-to-decoder feature consistency
loss_encoder1_1 = F.mse_loss(se_encoder_feature1, re_decoder_feature1)
loss_encoder2_1 = F.mse_loss(se_encoder_feature2, re_decoder_feature2)
loss_encoder3_1 = F.mse_loss(se_encoder_feature3, re_decoder_feature3)

# Decoder-to-encoder feature consistency
loss_decoder1_1 = F.mse_loss(se_decoder_feature1, re_encoder_feature1)
loss_decoder2_1 = F.mse_loss(se_decoder_feature2, re_encoder_feature2)
loss_decoder3_1 = F.mse_loss(se_decoder_feature3, re_encoder_feature3)

# Mid-level feature consistency
loss_midfeature = F.mse_loss(outputs[1], outputs2[1])

🎯 Total Optimization Loss
The final optimization loss combines cross-entropy, Dice loss, and reconstruction-guided feature consistency:


# Cross-entropy and Dice losses for segmentation output
loss_ce = ce_loss(outputs[0], label_batch.long())
loss_dice = dice_loss(outputs[0], label_batch, softmax=True)

# Feature consistency loss
loss_opt = (
    loss_decoder1_1 + loss_decoder2_1 + loss_decoder3_1 +
    loss_encoder1_1 + loss_encoder2_1 + loss_encoder3_1 +
    loss_midfeature
) / 3

# Total loss
a = 0.035  # weighting factor for consistency loss
loss = 0.5 * loss_ce + 0.5 * loss_dice + a * loss_opt

🧠 Intuition
By aligning internal representations between segmentation and reconstruction branches, the model learns more robust and generalizable features, leading to improved segmentation performance — especially in data-scarce or noisy scenarios.
