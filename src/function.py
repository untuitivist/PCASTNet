import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


def confusion_matrix(y_true, y_pred, labels):
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        if true in label_to_idx and pred in label_to_idx:
            cm[label_to_idx[true], label_to_idx[pred]] += 1
    return cm


def macro_f1_from_confusion(cm):
    scores = []
    for idx in range(cm.shape[0]):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def plot_confusion_matrix(cm, display_labels):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(display_labels)),
        yticks=np.arange(len(display_labels)),
        xticklabels=display_labels,
        yticklabels=display_labels,
        ylabel="True label",
        xlabel="Predicted label",
    )
    threshold = cm.max() / 2.0 if cm.size and cm.max() else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                format(cm[row, col], "d"),
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
            )
    fig.tight_layout()
    return fig, ax


def train_transform():
    transform_list = [
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
    ]
    return transforms.Compose(transform_list)


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def adjust_learning_rate1(lr, lr_decay, optimizer, iteration_count):
    """Imitating the original implementation."""
    lr = lr / (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def adjust_learning_rate2(lr, lr_decay, optimizer, iteration_count, step_size=10, gamma=0.5):
    """Step learning-rate decay."""
    lr = lr * (gamma ** (iteration_count // step_size))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def test_and_save_confusion_matrix(model, num_classes, test_loader, device, save_dir, i, label):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, total=len(test_loader), desc="Processing test segments", ncols=100, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = (all_preds == all_labels).sum() * 100 / len(all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    f1 = macro_f1_from_confusion(cm)
    try:
        all_labels_bin = label_binarize(all_labels, classes=range(num_classes))
        auc = roc_auc_score(all_labels_bin, all_probs, average="macro", multi_class="ovr")
    except ValueError:
        auc = float("nan")

    fig, ax = plot_confusion_matrix(cm, display_labels=range(num_classes))
    ax.set_title(f"Confusion Matrix {label}\nAccuracy: {accuracy:.2f}%, F1: {f1:.2f}, AUC: {auc:.2f}", pad=14)
    fig.tight_layout()
    fig.savefig(save_dir / f"{i:0>3}_confusion_matrix_{label}.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    np.save(save_dir / f"{i:0>3}_confusion_matrix_{label}.npy", cm)
    np.savez(
        save_dir / f"{i:0>3}_predictions_{label}.npz",
        labels=all_labels,
        preds=all_preds,
        probs=all_probs,
        accuracy=accuracy,
        f1=f1,
        auc=auc,
    )

    return accuracy


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 4
    n, c = size[:2]
    feat_var = feat.view(n, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(n, c, 1, 1)
    feat_mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert content_feat.size()[:2] == style_feat.size()[:2]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    assert feat.size()[0] == 3
    assert isinstance(feat, torch.FloatTensor)
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    u, d, v = torch.svd(x)
    return torch.mm(torch.mm(u, d.pow(0.5).diag()), v.t())


def coral(source, target):
    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)), source_f_norm),
    )
    source_f_transfer = source_f_norm_transfer * target_f_std.expand_as(source_f_norm) + target_f_mean.expand_as(source_f_norm)
    return source_f_transfer.view(source.size())


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


def reconstruct_gray_image(rgb_transformed_img, vmin=0, vmax=255):
    transformed = rgb_transformed_img
    if isinstance(transformed, torch.Tensor):
        transformed = transformed.detach().cpu().numpy()
        if transformed.ndim == 3:
            transformed = transformed.transpose(1, 2, 0)

    transformed = (transformed * 255).astype(np.uint8)
    transformed_pil = Image.fromarray(transformed)
    pixels = np.array(transformed_pil) / 255.0

    cmap = plt.get_cmap("jet")
    norm = plt.Normalize(vmin, vmax)
    values = np.linspace(vmin, vmax, 256)
    colors = cmap(norm(values))[:, :3]

    pixels_flat = pixels.reshape(-1, 3)
    diff = np.linalg.norm(pixels_flat[:, None] - colors[None, :], axis=2)
    closest_indices = np.argmin(diff, axis=1)
    data_values = values[closest_indices]
    gray_data = (data_values - vmin) / (vmax - vmin)
    return gray_data.reshape(pixels.shape[:2])


def batch_reconstruct_gray(rgb_batch, vmin=0, vmax=255):
    device = rgb_batch.device
    b, c, h, w = rgb_batch.shape

    if not hasattr(batch_reconstruct_gray, "colors_cache"):
        cmap = plt.get_cmap("jet")
        values = torch.linspace(vmin, vmax, 256, device=device)
        colors = cmap(plt.Normalize(vmin, vmax)(values.cpu().numpy()))[:, :3]
        batch_reconstruct_gray.colors_cache = torch.tensor(colors, device=device, dtype=torch.float32)
    colors = batch_reconstruct_gray.colors_cache

    if c == 1:
        pixels = rgb_batch.expand(-1, 3, -1, -1).permute(0, 2, 3, 1)
    else:
        pixels = rgb_batch.permute(0, 2, 3, 1)

    pixels = pixels.clamp(0, 1).reshape(b, -1, 3)
    diff = torch.cdist(pixels, colors.unsqueeze(0), p=2)
    _, closest_idx = torch.min(diff, dim=2)
    gray_values = torch.linspace(vmin, vmax, 256, device=device)[closest_idx]
    gray_data = (gray_values - vmin) / (vmax - vmin)
    return gray_data.view(b, h, w)
