from __future__ import annotations

from tensorboardX import SummaryWriter

from stc_utils import suffixed_run_dir, write_json


def prepare_stage_output(stc, save_dir, stage_suffix: str, setting: dict, indent: int | None = None):
    resolved_save_dir = suffixed_run_dir(save_dir, stc.model_name, stage_suffix)
    resolved_save_dir.mkdir(exist_ok=True, parents=True)
    log_dir = resolved_save_dir
    writer = SummaryWriter(log_dir=str(log_dir))
    print('[+]Log directory:', log_dir, 'Save directory:', resolved_save_dir, 'Logging...')
    setting = dict(setting)
    setting['save_dir'] = str(resolved_save_dir)
    setting['log_dir'] = str(log_dir)
    write_json(resolved_save_dir / 'setting.json', setting, indent=indent)
    print('[+]Save setting info', resolved_save_dir / 'setting.json')
    return resolved_save_dir, log_dir, writer


def freeze_module(module, label: str) -> None:
    for param in module.parameters():
        param.requires_grad = False
    print(f'[+]Freeze {label}')
