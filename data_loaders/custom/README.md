# Working with Custom Rigs

The [Flexible Motion In-Betweening][condmdi] model is trained on the [HumanML3D dataset][hml3d_fork],
originally by [Eric Guo][hml3d_orig], which is a combination of various motion-capture sequences, all
using the SMPL+ 22-node data structure. In order to train on a custom rig, we must specify the joints
of the rig, and edit where the assumptions are made in the training script.

This is the original workflow to obtain the HumanML3D dataset, summarized from the README there:

## Original Workflow for HumanML3D:
1. Download the various datasets from [AMASS][amass] then unzip them into the `amass_data/` folder in
   the HumanML3D repository. Next, download `SMPL+H` models from [MANO][mano] and `DMPLS` models from
   the [SMPL][smpl] sites. Unzip these and put them in the `body_models/` folder. Each of these sites
   requires an account to be created before you download anything. 
2. Run `raw_pose_preprocess.ipynb` on the data. This gets poses from the AMASS data.
3. Run the absolute value versions of `motion_processing.ipynb` and `cal_mean_variance.ipynb`. If you
   cloned the [original][hml3d_orig] repo, please copy the notebooks from the `HumanML3D_abs/` folder
   in [CondMDI][condmdi] to the root of the HumanML3D repo, then run those. In the [fork][hml3d_fork]
   the notebooks are the absolute root joint versions; the original notebooks have the prefix `rel_`.
4. Copy the processed data directory `HumanML3D/` into `dataset/`. The sequence data can now be found
   in `new_joints_abs_3d/`, with the converted data in `new_joint_vecs_abs_3d/`.

[amass]:        https://amass.is.tue.mpg.de/download.php
[smpl]:         https://smpl.is.tue.mpg.de/download.php
[mano]:         https://mano.is.tue.mpg.de/download.php
[condmdi]:      https://github.com/icedwater/CondMDI
[hml3d_fork]:   https://github.com/icedwater/HumanML3D
[hml3d_orig]:   https://github.com/EricGuo5513/HumanML3D

## Preparing a custom dataset for training

Make sure a corresponding set of `$DATASET/joints` and `$DATASET/vecs` is present.
The dimensions of each sequence nd-array in `joints` should be F x J x 3, F is the
number of frames, J the number of joints in the rig, and 3 the coordinates of each
joint. The `vecs` arrays should have dimensions F x (12J - 1) as per Appendix A of
the [paper][condpaper].

Each sequence must be accompanied by a text file containing some captions with the
following format:

    caption#tokens#from_tag#to_tag

where `caption` describes one action in the sequence, `tokens` is the caption in a
tokenised form, and the part of the sequence described by the caption is delimited
by `from_tag` and `to_tag`. These last two values may be 0.0, in which case all of
the sequence is used. [^confirm this] In the open data, mirrored captions are kept
in the files beginning with `M`:

    $ cat 003157.txt (truncated)
    a person makes a turn to the right.#a/DET person/NOUN make/VERB a/DET turn/VERB to/ADP the/DET right/NOUN#0.0#0.0
    $ cat M003157.txt (truncated)
    a person makes a turn to the left.#a/DET person/NOUN make/VERB a/DET turn/VERB to/ADP the/DET left/NOUN#0.0#0.0

This can be generated separately, but is done by step 3 in the original workflow.

[condpaper]:    https://arxiv.org/html/2405.11126v2#A1

- How to recalculate `mean` and `std` from the original? 

This is a summary of the steps to train on a custom rig called "myrig":

1. Copy the `data_loaders/custom` directory to `data_loaders/myrig`.
2. Update the dataset info for `myrig` in `data_loaders/myrig/data/dataset.py`.
3. Update `data_loaders/get_data.py`.
4. Update `utils/model_util.py`.
  - create_model_and_diffusion is here: assumes unet, loads data
  - add class name and specs to get_model_args
- model/mdm_unet.py:
  - in MDM_UNET class:
    - add class name and added_channels to __init__ part
    - add class name and parameters to encode_text()
    - add class name to assertions in forward()
    - add class name and njoints to forward_core()
- utils/get_opt:
  - create custom dataset_name in get_opt: allow for new settings
- add custom to main() in sample.conditional_synthesis; assertions fail otherwise
- scripts/motion_process
  - update coords (upper legs, feet, face vectors, hips) and joints_num

-- train_condmdi::main()
  - train_args(base_cls=card.motion_abs_unet_adagn_xl) <-- overwrite card here? or just leave it
    - card inherits configs/data/dataset name ("humanml")

The details of each step are highlighted below.

### Create a new data_loader class called `myrig`

Copy the `data_loaders/custom` directory to a new directory, `data_loaders/myrig`.

### Update dataset info for `myrig` in `data_loaders/myrig/data/dataset.py`

This file contains the specific settings for this rig.

  - create new subclass from data.Dataset here with specific settings
    - /dataset/humanml_opt.txt is loaded as `opt` and `self.opt` within subclass
  - import necessary dependencies (ignore t2m?)
  - Text2MotionDatasetV2 and TextOnlyDataset depend on `--joints_num`, include that
  - train t2m for custom rig here
    - min_motion_len = 40 for t2m, else 24 (sequences below 24/40 frames are skipped)
  - update the feet and facing joints in `motion_to_rel_data` and `motion_to_abs_data`
    - start and end joints of left foot and right foot
    - facing joints are Z-shape: right hip, left hip, right shoulder, left shoulder
  - update the njoints in `sample_to_motion`, `abs3d_to_rel`, and `rel_to_abs3d`
    - 22 is the default value for the HumanML3D dataset.

### Update data_loaders/get_data.py

  - add `myrig` to the list of valid classes in `get_dataset_class` and `get_dataset`
  - update `get_collate_fn()`; how does collate function apply here? need to make own?
  - get_model_args() should have the correct njoints
  - data_rep needs to be updated

(...)

- get_dataset_class
- get_dataset
- get_collate_fn


## Using the trained custom model for inference

```bash
python -m sample.conditional_synthesis --dataset="custom" ...
```

----
> end of document
----

# Working Notes to Explore

- can the existing scripts convert arbitrary J-joint rigs to the correct form?
- will need to update momask joints2bvh: convert() to use nonstandard rig as well
- where is max_motion_length (=224?) set and what does it control
- what is the Pointer? (max_length = 20? in reset_max_len: self.pointer = np.searchsorted(self.length_arr, length))
- explain create_model_and_diffusion
- where is "lantent dim"?
- what happens in rot2xyz.smpl_model.eval() and is it necessary?

## Current training output

- creating data loader...
  - data <== get_dataset_loader(data_conf)
  - data_conf <== DatasetConfig(dataset="humanml", batch_size=64, num_frames=60*, abs_3d=False, traj_only=False, use_random_proj=False, random_proj_scale=10.0, augment_type="none", std_scale_shift=(1.0, 0.0), drop_redundant=False)

- ././dataset/humanml_opt.txt
- WARNING: max_motion_length is set to 224
- Loading dataset t2m ... / mode = train
- t2m dataset aug: none std_scale_shift: (1.0, 0.0) drop redundant information: False (...23384?)
- Pointer Pointing at 0
- creating model and diffusion...
  - model, diffusion = cmd(args, data)
    - xxx
  - model.rot2xyz.smpl_model.eval() <--- what happens here?
- using UNET with lantent dim: 512 and mults: (2, 2, 2, 2)
- dims: [263, 1024, 1024, 1024, 1024] mults: (2, 2, 2, 2)
- [ models/temporal ] Channel dimensions: [(263, 1024), (1024, 1024), (1024, 1024), (1024, 1024)]
- EMBED TEXT
- Loading CLIP...
- Total params: 235.12M (doesn't seem to change)
  - sum(p.numel() for p in model.parameters_wo_clip()) <-- what xcomp is this, where is pwoclip()? 

## Data_Loader/...

## Get_Data.py
- dataset:
  - add new type in DataOptions
  - where is keyframe_conditioned used in ModelOptions?
  - get_data.py:
    - get_dataset_class("classname")
    - from data_loaders.CLASSNAME.data import CLASSNAME

## From the Training Arguments dataclasses
args = train_args(base_cls=card.motion_abs_unet_adagn_xl)
     = HfArgumentParser(base_cls).parse_args_into_dataclasses()[0]

--> does the base_cls affect any params?

--> TrainArgs(BaseOptions, DataOptions, ModelOptions, DiffusionOptions, TrainingOptions)
    - cuda: bool=True
    - device: int=0
    - seed: int=10

    - dataset: str="humanml", ["humanml", "kit", "humanact12", "uestc", "amass"]
    - data_dir: str="", check dataset defaults
    - abs_3d: bool=False
    - traj_only: bool=False
    - xz_only: Tuple[int]=False??
    - use_random_proj: bool=False
    - random_proj_scale: float=10.0
    - augment_type: str="none", ["none", "rot", "full"]
    - std_scale_shift: Tuple[float]=(1.0, 0.0)
    - drop_redundant: bool=False (if true, keep only 4 + 21*3)

    - arch: str="trans_enc", check paper for arch types
    - emb_trans_dec: bool=False (toggle inject condition as class token in trans_dec)
    - layers: int=8
    - latent_dim: int=512 (tf/gru width)
    - ff_size: int=1024 (tf feedforward size)
    - dim_mults: Tuple[float]=(2, 2, 2, 2) (channel multipliers for unet)
    - unet_adagn: bool=True (adaptive group normalization for unet)
    - unet_zero: bool=True (zero weight init for unet)
    - out_mult: bool=1 (large variation feature multiplier for unet/tf)
    - cond_mask_prob: float=0.1 (prob(mask cond during training) for cfg)
    - keyframe_mask_prob: float=0.1 (prob(mask keyframe cond during training) for cfg)
    - lambda_rcxyz: float=0.0, joint pos loss
    - lambda_vel: float=0.0, joint vel loss
    - lambda_fc: float=0.0, foot contact loss
    - unconstrained: bool=False (training independent of text, action. only humanact12)
    - keyframe_conditioned: bool=False (condition on keyframes. only hml3d)
    - keyframe_selection_scheme: str="random_frames", ["random_frames", "random_joints", "random"]
    - zero_keyframe_loss: bool=False (zero the loss over observed keyframe loss, or allow model to make predictions over observed keyframes if false)

    - noise_schedule: str="cosine"
    - diffusion_steps: int=1000, T in paper
    - sigma_small: bool=True, what?
    - predict_xstart: bool=True, what?
    - use_ddim: bool=False, what?
    - clip_range: float=6.0, range to clip what?

    - save_dir: str=None
    - overwrite: bool=False, true to reuse existing dir
    - batch_size: int=64
    - train_platform_type: str="NoPlatform", ["NoPlatform", "ClearmlPlatform", "TensorboardPlatform", "WandbPlatform"]
    - lr: float=1e-4, learning rate
    - weight_decay: float=0, optimizer weight decay
    - grad_clip: float=0, gradient clip
    - use_fp16: bool=False
    - avg_model_beta: float=0, 0 means disabled
    - adam_beta2: float=0.999
    - lr_anneal_steps: int=0
    - eval_batch_size: int=32, <<do not touch warning>>
    - eval_split: str="test", ["val", "test"]
    - eval_during_training: bool=False
    - eval_rep_times: int=3, times to loop evaluation during training
    - eval_num_samples: int=1_000, set to -1 to use all
    - log_interval: int=1_000, N steps before losses should be logged
    - save_interval: int=100_000, N steps to save checkpoint AND run evaluation if asked
    - num_steps: int=1_200_000
    - num_frames: int=60, frame limit ignored by hml3d and KIT (check what the value there is)
    - resume_checkpoint: str="", continue training from checkpoint 'model_.pt'
    - apply_zero_mask: bool=False
    - traj_extra_weight: float=1.0, extra weight for what?
    - time_weighted_loss: bool=False, what does this do?
    - train_x0_as_eps: bool=False, what is x0 and what is eps?


### benchmark-sparse issue: https://github.com/setarehc/diffusion-motion-inbetweening/issues/5#issuecomment-2197243178

With `--edit_mode` set to `benchmark_sparse` and `transition_length` set to 5, keyframes are being defined every 5 frames.
This creates a very strong keyframe condition that leaves little room for the text condition to influence the results.
For optimal use of text conditioning, it's better to condition on specific joint trajectories. This way, you can control
these joints while allowing the text prompt to guide the movements of the other, free joints.

You can also increase the classifier-free sampling weight for text conditioning by setting `guidance_param` to values higher
than 2.5 to increase the effect of text.

## params

- `--keyframe_guidance_param`: 1 but what else is possible
- `--keyframe_selection_scheme`: `random_joints` but what else is possible

### reading the edit_args JSON produced at inference time training

    "adam_beta2": 0.999,

{
    "abs_3d": true,
    "action_file": "",
    "action_name": "",
    "apply_zero_mask": false,
    "arch": "unet",
    "augment_type": "none",
    "avg_model_beta": 0.9999,
    "batch_size": 1,
    "clip_range": 6.0,
    "cond_mask_prob": 0.1,
    "cuda": true,
    "cutoff_point": 0,
    "data_dir": "",
    "dataset": "humanml",
    "device": 0,
    "diffusion_steps": 1000,
    "dim_mults": [
        2,
        2,
        2,
        2
    ],
    "drop_redundant": false,
    "edit_mode": "benchmark_sparse",
    "editable_features": "pos_rot_vel",
    "emb_trans_dec": false,
    "eval_batch_size": 32,
    "eval_during_training": false,
    "eval_num_samples": 1000,
    "eval_rep_times": 3,
    "eval_split": "test",
    "ff_size": 1024,
    "grad_clip": 1.0,
    "gradient_schedule": null,
    "guidance_param": 2.5,
    "imputate": false,
    "input_text": "",
    "keyframe_conditioned": true,
    "keyframe_guidance_param": 1.0,
    "keyframe_mask_prob": 0.1,
    "keyframe_selection_scheme": "random_joints",
    "lambda_fc": 0.0,
    "lambda_rcxyz": 0.0,
    "lambda_vel": 0.0,
    "latent_dim": 512,
    "layers": 8,
    "log_interval": 1000,
    "lr": 0.0001,
    "lr_anneal_steps": 0,
    "model_path": "./save/randomframes/model000750000.pt",
    "motion_length": 11.2,
    "motion_length_cut": 6.0,
    "n_keyframes": 5,
    "no_text": false,
    "noise_schedule": "cosine",
    "num_frames": 224,
    "num_repetitions": 3,
    "num_samples": 1,
    "num_steps": 3000000,
    "out_mult": false,
    "output_dir": "",
    "overwrite": false,
    "predict_xstart": true,
    "random_proj_scale": 10.0,
    "reconstruction_guidance": false,
    "reconstruction_weight": 5.0,
    "replacement_distribution": "conditional",
    "resume_checkpoint": "",
    "save_dir": "save/snjua2bq",
    "save_interval": 50000,
    "seed": 10,
    "sigma_small": true,
    "std_scale_shift": [
        1.0,
        0.0
    ],
    "stop_imputation_at": 0,
    "stop_recguidance_at": 0,
    "text_condition": "",
    "text_prompt": "roundhouse kick",
    "time_weighted_loss": false,
    "train_platform_type": "NoPlatform",
    "train_x0_as_eps": false,
    "traj_extra_weight": 1.0,
    "traj_only": false,
    "transition_length": 100,
    "unconstrained": false,
    "unet_adagn": true,
    "unet_zero": true,
    "use_ddim": false,
    "use_fixed_dataset": false,
    "use_fixed_subset": false,
    "use_fp16": true,
    "use_random_proj": false,
    "weight_decay": 0.01,
    "xz_only": false,
    "zero_keyframe_loss": false
}

