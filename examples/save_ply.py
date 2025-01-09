from pathlib import Path

import yaml
from simple_trainer import Config, Runner
import torch
import os
import numpy as np
from plyfile import PlyData, PlyElement
import tyro

# Experimental
def construct_list_of_attributes(runner):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(runner.splats["sh0"].shape[1]*runner.splats["sh0"].shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(runner.splats["shN"].shape[1]*runner.splats["shN"].shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(runner.splats["scales"].shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(runner.splats["quats"].shape[1]):
        l.append('rot_{}'.format(i))
    return l

# Experimental
@torch.no_grad()
def save_ply(runner, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("Saving to", path)

    xyz = runner.splats["means"].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = runner.splats["sh0"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = runner.splats["shN"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = runner.splats["opacities"].detach().unsqueeze(-1).cpu().numpy()
    scale = runner.splats["scales"].detach().cpu().numpy()
    rotation = runner.splats["quats"].detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(runner)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    args = parser.parse_args()

    # Load the model
    cfg_path = args.input / "cfg.yml"
    cfg_dict : dict = yaml.load(cfg_path.read_text(), yaml.Loader)
    # assign all attributes to a config object
    cfg: Config = Config(**cfg_dict)
    # cfg = tyro.load_cfg(cfg_path)
    runner = Runner(0, 0, 1, cfg)

    ckpt = torch.load(args.input / "ckpts" / "ckpt_29999_rank0.pt", map_location=runner.device, weights_only=True)
    for k in runner.splats.keys():
        runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in [ckpt]])

    save_ply(runner, args.input / (args.input.name + ".ply"))