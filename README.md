# Temporally and Spatially Novel Video Frame Synthesis using 4D Video Autoencoder

Repository for the Stanford CS231N project: "Temporally and Spatially Novel Video Frame Synthesis using 4D Video Autoencoder." The repository includes implementation for the flow-enabled 4D Video Autoencoder, training scripts and test scripts. The code was adapted from [zlai0/VideoAutoencoder](https://github.com/zlai0/VideoAutoencoder/) of "Video Autoencoder: self-supervised disentanglement of 3D structure and motion" by Lai et al. ([paper](https://arxiv.org/abs/2110.02951)). Citation for this work can be found below:
```
@inproceedings{Lai21a,
        title={Video Autoencoder: self-supervised disentanglement of 3D structure and motion},
        author={Lai, Zihang and Liu, Sifei and Efros, Alexei A and Wang, Xiaolong},
        booktitle={ICCV},
        year={2021}
}
```

## Prerequisites
Install dependences following the instructions [here](https://github.com/KathyFeiyang/cs231n-project/tree/main/VideoAutoencoder#dependencies). These prerequisites are the same as the original Video Autoencoder.

## Usage
`flow_train.py` is the entry point for training the 4D Video Autoencoder. A sample command is:
```
python flow_train.py --savepath log/model --resume mp3d.ckpt --dataset HMDB51 --epochs 100 --interval 1 --lr 2e-4 --bsize 1 --clip_length 4
```

`flow_test_interpolate.py` is the entry point for interpolating novel middle frames. A sample command is:
```
python flow_test_interpolate.py --savepath log/exp0_inter --resume flow_checkpoint.tar --dataset HMDB51 --interval 1 --video_limit 1000
```

After generating videos frames, such as through interpolation, you can use `eval_syn_re10k.py` to evaluate the quality of the generations using LIPIS, PSNR and SSIM. A sample command is:
```
python eval_syn_re10k.py --lpips log/interpolation_output/Videos/
```

## Model checkpoint
We release the model checkpoint from our best performing model [here](https://drive.google.com/file/d/1l2uG2mx2O836f827Go7fCXVB4zaJekM9/view?usp=sharing).

## License
This repository is made available to the public under the MIT license.
