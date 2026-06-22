from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from function import test_and_save_confusion_matrix
from stc_utils import update_json


def save_classifier_checkpoint(stc, train_loader, val_loader, test_loader, device, save_dir, iteration, label):
    net = stc.net
    net.eval()
    test_acc = test_and_save_confusion_matrix(
        net, stc.num_classes, test_loader, device, save_dir, iteration, f'{label}_test'
    )
    print(f'[+]Test Accuracy: {test_acc:.2f}%')
    print(f' - Confusionmatrix saved at: {save_dir / f"{iteration:0>3}_confusion_matrix_{label}_[train, valid, test].png"}')
    save_model_pth = f"{save_dir}/{iteration:0>3}_iter_{label}.pth.tar"
    net.save(save_model_pth)
    net.train()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.dim() == 4:
        tensor = make_grid(tensor)
    tensor = tensor.detach().cpu().clamp(0, 1)
    ndarr = (tensor * 255).add_(0.5).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(ndarr)


def save_style_transfer_checkpoint(stc, content_test_images, style_test_images, save_dir, iteration, label):
    net = stc.net
    net.eval()
    device = stc.device

    total_loss_c = 0
    total_loss_s = 0
    total_loss_p = 0
    total_loss_v = 0
    total_loss_e = 0
    generated = []

    with torch.no_grad():
        for content_image in tqdm(content_test_images, desc=" - Processing Content Images"):
            content_images = content_image.unsqueeze(0).to(device)
            generated_for_content = []
            for style_image in tqdm(style_test_images, desc=" - Processing Style Images", leave=False):
                style_images = style_image.unsqueeze(0).to(device)
                g_t, loss_c, loss_s, loss_p, loss_v, loss_e = net(content_images, style_images)
                total_loss_c += loss_c
                total_loss_s += loss_s
                total_loss_p += loss_p
                total_loss_v += loss_v
                total_loss_e += loss_e
                generated_for_content.append(g_t.cpu())
            generated.append(generated_for_content)

    pair_count = len(content_test_images) * len(style_test_images)
    loss_c = total_loss_c / pair_count
    loss_s = total_loss_s / pair_count
    loss_p = total_loss_p / pair_count
    loss_v = total_loss_v / pair_count
    loss_e = total_loss_e / pair_count

    img_size = tensor_to_pil(content_images[0]).size[0]
    grid_size = ((len(style_test_images) + 1) * img_size, (len(content_test_images) + 1) * img_size)
    grid_image = Image.new("RGB", grid_size, (255, 255, 255, 0))

    for i in range(len(style_test_images)):
        grid_image.paste(tensor_to_pil(style_test_images[i]), ((i + 1) * img_size, 0))
    for i in range(len(content_test_images)):
        grid_image.paste(tensor_to_pil(content_test_images[i]), (0, (i + 1) * img_size))
    for i in range(len(content_test_images)):
        for j in range(len(style_test_images)):
            grid_image.paste(tensor_to_pil(generated[i][j]), ((j + 1) * img_size, (i + 1) * img_size))

    print(f' - Test Content Loss: {loss_c.item():.4f},\tTest Style Loss: {loss_s.item():.4f},\tTest Perceptual Loss: {loss_p.item():.4f},\tTest Total Variation Loss: {loss_v.item():.4f},\tTest Entropy Loss: {loss_e.item():.4f}')
    grid_image.save(f'{save_dir}/{iteration:0>3}_iter_{label}.png')
    print(f' - Style transfer collage image saved at: {save_dir / f"{iteration:0>3}_iter_{label}.png"}')

    update_json(save_dir / 'setting.json', {
        'test_content_loss': loss_c.item(),
        'test_style_loss': loss_s.item(),
        'test_perceptual_loss': loss_p.item(),
        'test_total_variation_loss': loss_v.item(),
        'test_entropy_loss': loss_e.item(),
    })
    net.train()


def save_feature_map(feature, save_path_prefix):
    feature = feature.squeeze(0)
    for i in range(min(3, feature.size(0))):
        fmap = feature[i].detach().cpu()
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-6)
        save_image(fmap.unsqueeze(0), f"{save_path_prefix}_{i}.png")


def save_feature_maps(stc, contents, styles, save_dir, tag, prefix):
    stc.net.eval()
    save_dir = Path(save_dir)
    content = contents[0].unsqueeze(0).to(stc.device)
    style = styles[0].unsqueeze(0).to(stc.device)

    with torch.no_grad():
        content_feat = stc.net.encode(content)
        style_feat = stc.net.encode(style)
        transfer_feat = stc.net.adailn(content_feat, style_feat)
        transfer = stc.net.decoder(transfer_feat)

        save_image(content, save_dir / f'{prefix}_content.png')
        save_image(style, save_dir / f'{prefix}_style.png')
        save_image(transfer, save_dir / f'{prefix}_transfer.png')

        save_feature_map(content_feat, save_dir / f'{prefix}_content_feature_map')
        save_feature_map(style_feat, save_dir / f'{prefix}_style_feature_map')
        save_feature_map(transfer_feat, save_dir / f'{prefix}_transfer_feature_map')
