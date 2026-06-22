from __future__ import annotations

import json
import os
import time

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from data_loader import CustomImageDataset


def make_adailn_dataset(stc, folder_label):
    if stc.net is None:
        raise ValueError('[!]Please load the network first!')
    net = stc.net
    net.to(stc.device)
    net.eval()
    class_list = list(range(stc.num_classes))
    style_transfer_num_total = int(stc.content_train_dataset_scale)
    add_num_per_class = int(style_transfer_num_total / stc.num_classes)
    style_num_per_class = int(stc.style_train_dataset_scale / stc.num_classes)
    content_num_per_class = int(add_num_per_class / style_num_per_class)
    print('[+]Content dataset scale:', stc.content_train_dataset_scale)
    print('[+]Style dataset scale:', stc.style_train_dataset_scale)
    print('[+]Need style transfer dataset scale:', style_transfer_num_total)
    print('[+]Need style transfer dataset scale per class:', style_num_per_class)
    print('[+]Need content transfer data scale per class:', content_num_per_class)
    print('[+]Need style transfer data scale per class:', style_num_per_class)
    print('[+]Load content data from', stc.content_dataset_dir)
    content_train_dataset = CustomImageDataset(
        stc.content_dataset_dir + '/train',
        transform=stc.test_content_tf,
        scale=int(stc.content_train_dataset_scale / stc.num_classes),
    )
    print('[+]Load style data from', stc.style_dataset_dir)
    style_train_dataset = CustomImageDataset(
        stc.style_dataset_dir + '/train',
        transform=stc.test_style_tf,
        scale=int(stc.style_train_dataset_scale / stc.num_classes),
    )
    class_style_content_dict = {}
    root_folder = stc.style_transfer_dataset_dir + '/train'
    os.makedirs(root_folder, exist_ok=True)
    for i in class_list:
        class_style_content_dict[i] = {
            'style': style_train_dataset.class_samples(i, style_num_per_class),
            'content': content_train_dataset.class_samples(i, content_num_per_class),
        }
    for i in class_style_content_dict:
        print('Class', i, ':', len(class_style_content_dict[i]['style']), 'style images,', len(class_style_content_dict[i]['content']), 'content images')
        index = 0
        for content_image, content_img_path, content_label in tqdm(class_style_content_dict[i]['content'], leave=False):
            folder_path = os.path.join(root_folder, content_label + '_' + folder_label)
            os.makedirs(folder_path, exist_ok=True)
            content = content_image.to(stc.device).unsqueeze(0)
            for style_image, style_img_path, style_label in tqdm(class_style_content_dict[i]['style'], leave=False):
                if style_label == content_label:
                    style = style_image.to(stc.device).unsqueeze(0)
                    output_path = os.path.join(f'{folder_path}/{content_label}_{index}_{content_img_path.split("_")[-1].split(".")[0]}-{style_img_path.split("_")[-1]}')
                    with torch.no_grad():
                        output, _, _, _, _, _ = net(content, style)
                    save_image(output.cpu(), output_path)
                    index += 1
                else:
                    print('[!]Style label:', style_label, 'Content label:', content_label)
    print('[+]Save style transfer images to', root_folder)

    dataset_info = {
        'model_name': stc.model_name,
        'num_classes': stc.num_classes,
        'content_dataset_dir': stc.content_dataset_dir,
        'style_dataset_dir': stc.style_dataset_dir,
        'style_transfer_dataset_dir': stc.style_transfer_dataset_dir,
        'content_train_dataset_scale': stc.content_train_dataset_scale,
        'style_train_dataset_scale': stc.style_train_dataset_scale,
        'folder_label': folder_label,
        'generation_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    info_path = os.path.join(root_folder, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=4)
    print('[+]Dataset info saved to', info_path)
