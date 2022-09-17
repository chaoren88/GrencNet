import glob, cv2, os
import numpy as np
from skimage import io, img_as_ubyte
from tcsvt.GrancNet import DN
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load('checkpoint_39.45_0.9188.pth', map_location=device)
model = DN().to(device)
model.load_state_dict(checkpoint)
model.eval()

image_list = glob.glob('./dataset/SIDD_noise' + "/*")

save_denoised_image = False

psnr = 0
ssim = 0

for i in range(len(image_list)):
    image_name = image_list[i]
    print('Image: {:02d}, path: {:s}'.format(i+1, image_name))
    path_label = os.path.split(image_name)
    gt_name = path_label[1].split('.')[0].replace('_noise','')
    im_input = io.imread(image_name)
    im_input = im_input
    im_input = np.transpose(im_input, axes=[2, 0, 1]).astype('float32') / 255.0
    noisy_img = torch.from_numpy(im_input).to(device)
    noisy_img = noisy_img.unsqueeze(0)
    with torch.no_grad():
        aa, test_out = model(noisy_img)

    im_denoise = test_out.data
    im_denoise.clamp_(0.0, 1.0)
    im_denoise = img_as_ubyte(im_denoise.squeeze(0).numpy().transpose([1, 2, 0]))
    print(im_denoise)

    gt = io.imread('./dataset/SIDD_clean/' + gt_name + '.png')
    psnr_iter = compare_psnr(im_denoise, gt, data_range=255)
    ssim_iter = compare_ssim(im_denoise, gt, data_range=255, gaussian_weights=True, use_sample_covariance=False,
                             multichannel=True)
    psnr += psnr_iter
    ssim += ssim_iter
    _, save_name = os.path.split(image_name)
    if save_denoised_image:
        io.imsave(os.path.join('./results', gt_name + '_{:.2f}_{:.4f}.png'.format(psnr_iter,ssim_iter)), im_denoise)
    exit(1)

print("///////////////////////////////////////")
print('PSNR: {:.2f}, SSIM: {:.4f}'.format(psnr,ssim))