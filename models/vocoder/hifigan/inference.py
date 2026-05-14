from __future__ import absolute_import, division, print_function, unicode_literals

import os
import json
import torch
from utils.util import AttrDict
from models.vocoder.hifigan.models import Generator

generator = None       # type: Generator
output_sample_rate = None     
_device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def load_model(weights_fpath, config_fpath=None, verbose=True):
    global generator, _device, output_sample_rate

    if verbose:
        print("Building hifigan")

    if config_fpath == None:
        model_config_fpaths = list(weights_fpath.parent.rglob("*.json"))
        if len(model_config_fpaths) > 0:
            config_fpath = model_config_fpaths[0]
        else:
            config_fpath = "./vocoder/hifigan/config_16k_.json"
    with open(config_fpath) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    output_sample_rate = h.sampling_rate
    torch.manual_seed(h.seed)

    if torch.cuda.is_available():
        # _model = _model.cuda()
        _device = torch.device('cuda')
    else:
        _device = torch.device('cpu')

    generator = Generator(h).to(_device)
    checkpoint = load_checkpoint(weights_fpath, _device)

    # Determine actual state_dict for generator in various checkpoint formats
    if isinstance(checkpoint, dict):
        if 'generator' in checkpoint and isinstance(checkpoint['generator'], dict):
            state_dict = checkpoint['generator']
        elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
            state_dict = checkpoint['state_dict']
        elif 'model_state' in checkpoint and isinstance(checkpoint['model_state'], dict):
            state_dict = checkpoint['model_state']
        else:
            # maybe the checkpoint is already the state dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Helper to strip common prefixes like 'module.' or 'generator.'
    def _strip_prefix(sd, prefix):
        keys = list(sd.keys())
        if any(k.startswith(prefix) for k in keys):
            return { (k[len(prefix):] if k.startswith(prefix) else k): v for k,v in sd.items() }
        return sd

    state_dict = _strip_prefix(state_dict, 'module.')
    state_dict = _strip_prefix(state_dict, 'generator.')

    # Load with strict=False to be tolerant to slight key mismatches
    try:
        load_result = generator.load_state_dict(state_dict, strict=False)
        if verbose:
            try:
                print('hifigan load_state_dict result: missing_keys=', load_result.missing_keys)
                print('hifigan load_state_dict result: unexpected_keys=', load_result.unexpected_keys)
            except Exception:
                print('hifigan load_state_dict result:', load_result)
    except Exception as exc:
        # fallback: try to load the checkpoint directly as a state_dict
        try:
            generator.load_state_dict(checkpoint)
        except Exception as exc2:
            print('Failed to load HiFi-GAN generator state_dict:', exc2)
            raise exc

    generator.eval()
    generator.remove_weight_norm()


def is_loaded():
    return generator is not None


def infer_waveform(mel, progress_callback=None):

    if generator is None:
        raise Exception("Please load hifi-gan in memory before using it")

    mel = torch.FloatTensor(mel).to(_device)
    mel = mel.unsqueeze(0)

    with torch.no_grad():
        y_g_hat = generator(mel)
        audio = y_g_hat.squeeze()
    audio = audio.cpu().numpy()

    return audio, output_sample_rate

