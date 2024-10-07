import shutil

from . import dump_json


def train_ckpt(instance):
    data = {attr: getattr(instance, attr) for attr in getattr(instance, 'attr_to_save')}

    args = getattr(instance, 'args')

    dump_json(data, f"{args['result_dir']}/train_ckpt_info.json")
    train_iter = getattr(instance, 'train_iter')
    if not train_iter % args['save_every']:
        shutil.copytree(args['ckpt_dir'], f"{args['result_dir']}/saved_train_ckpt/{train_iter}", dirs_exist_ok=True)
