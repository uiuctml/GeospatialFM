# patch_size = 16
# num_patches = 256
# perception_field_mask_256 = get_perception_field_mask(num_patches, patch_size, 10, attention_radius=640, cls_token=False)
# num_patches = 196
# perception_field_mask_196 = get_perception_field_mask(num_patches, patch_size, 10, attention_radius=640, cls_token=False)
# num_patches = 144
# perception_field_mask_144 = get_perception_field_mask(num_patches, patch_size, 10, attention_radius=640, cls_token=False)
# num_patches = 64
# perception_field_mask_64 = get_perception_field_mask(num_patches, patch_size, 10, attention_radius=640, cls_token=False)
# # find the median mask to visualize
# median_mask = 0.55
# vis_perception_mask_256 = perception_field_mask_256[int(256*median_mask)].reshape(int(math.sqrt(perception_field_mask_256.shape[1])), int(math.sqrt(perception_field_mask_256.shape[1]))).numpy()
# vis_perception_mask_196 = perception_field_mask_196[int(196*median_mask)].reshape(int(math.sqrt(perception_field_mask_196.shape[1])), int(math.sqrt(perception_field_mask_196.shape[1]))).numpy()
# vis_perception_mask_144 = perception_field_mask_144[int(144*median_mask)].reshape(int(math.sqrt(perception_field_mask_144.data.shape[1])), int(math.sqrt(perception_field_mask_144.data.shape[1]))).numpy()
# vis_perception_mask_64 = perception_field_mask_64[int(64*median_mask)].reshape(int(math.sqrt(perception_field_mask_64.shape[1])), int(math.sqrt(perception_field_mask_64.shape[1]))).numpy()

# # upsample the mask to the same size as the image
# vis_perception_mask_256 = TF.resize(torch.tensor(vis_perception_mask_256).unsqueeze(0), size=(256, 256), interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
# vis_perception_mask_196 = TF.resize(torch.tensor(vis_perception_mask_196).unsqueeze(0), size=(196, 196), interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
# vis_perception_mask_144 = TF.resize(torch.tensor(vis_perception_mask_144).unsqueeze(0), size=(144, 144), interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
# vis_perception_mask_64 = TF.resize(torch.tensor(vis_perception_mask_64).unsqueeze(0), size=(64, 64), interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

# img_256 = rgb_image.copy()
# img_196 = TF.center_crop(torch.tensor(img_256).permute(2, 0, 1), output_size=(196, 196)).permute(1, 2, 0).numpy()
# img_144 = TF.center_crop(torch.tensor(img_256).permute(2, 0, 1), output_size=(144, 144)).permute(1, 2, 0).numpy()
# img_64 = TF.center_crop(torch.tensor(img_256).permute(2, 0, 1), output_size=(64, 64)).permute(1, 2, 0).numpy()

# fig, ax = plt.subplots(2, 2, figsize=(10, 10))
# ax[0, 0].imshow(((1-vis_perception_mask_256.unsqueeze(-1).numpy()) * 0.7 + (1-vis_perception_mask_256.unsqueeze(-1).numpy()) * img_256 * 0.3 + (vis_perception_mask_256.unsqueeze(-1)).numpy()*img_256))
# # square the central patch
# # find the location of the target patch
# target_patch_idx = int(median_mask*256)
# target_i = target_patch_idx // 16
# target_j = target_patch_idx % 16
# ax[0, 0].add_patch(plt.Rectangle((target_j*16, target_i*16), 16, 16, edgecolor='red', facecolor='none'))
# ax[0, 0].set_title("Image Size: 256x256")
# ax[0, 0].axis('off')
# ax[0, 1].imshow(((1-vis_perception_mask_196.unsqueeze(-1).numpy()) * 0.7 + (1-vis_perception_mask_196.unsqueeze(-1).numpy()) * img_196 * 0.3 + (vis_perception_mask_196.unsqueeze(-1)).numpy()*img_196))
# ax[0, 1].set_title("Image Size: 196x196")
# target_patch_idx = int(median_mask*196)
# target_i = target_patch_idx // 14
# target_j = target_patch_idx % 14
# ax[0, 1].add_patch(plt.Rectangle((target_j*14, target_i*14), 14, 14, edgecolor='red', facecolor='none'))
# ax[0, 1].axis('off')

# ax[1, 0].imshow(((1-vis_perception_mask_144.unsqueeze(-1).numpy()) * 0.7 + (1-vis_perception_mask_144.unsqueeze(-1).numpy()) * img_144 * 0.3 + (vis_perception_mask_144.unsqueeze(-1)).numpy()*img_144))
# ax[1, 0].set_title("Image Size: 144x144")
# target_patch_idx = int(median_mask*144)
# target_i = target_patch_idx // 12
# target_j = target_patch_idx % 12
# ax[1, 0].add_patch(plt.Rectangle((target_j*12, target_i*12), 12, 12, edgecolor='red', facecolor='none'))
# ax[1, 0].axis('off')

# ax[1, 1].imshow(((1-vis_perception_mask_64.unsqueeze(-1).numpy()) * 0.7 + (1-vis_perception_mask_64.unsqueeze(-1).numpy()) * img_64 * 0.3 + (vis_perception_mask_64.unsqueeze(-1)).numpy()*img_64))
# ax[1, 1].set_title("Image Size: 64x64")
# target_patch_idx = int(median_mask*64)
# target_i = target_patch_idx // 8
# target_j = target_patch_idx % 8
# ax[1, 1].add_patch(plt.Rectangle((target_j*8, target_i*8), 8, 8, edgecolor='red', facecolor='none'))
# ax[1, 1].axis('off')
# plt.tight_layout()
# plt.show()