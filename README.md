# PyTorch-InstantNGP
PyTorch Implementation of NVIDIA's [Instant-NGP](https://github.com/NVlabs/instant-ngp)
```bibtex
@article{mueller2022instant,
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    journal = {ACM Trans. Graph.},
    issue_date = {July 2022},
    volume = {41},
    number = {4},
    month = jul,
    year = {2022},
    pages = {102:1--102:15},
    articleno = {102},
    numpages = {15},
    url = {https://doi.org/10.1145/3528223.3530127},
    doi = {10.1145/3528223.3530127},
    publisher = {ACM},
    address = {New York, NY, USA},
}
```
-------------

<img src="assets\nerf.gif" />

### Requirement

* Python 3.7
* Pytorch witch cuda (tested with 1.12.1)
* opencv
* imageio
* shutil
* tqdm
* nerfacc

### Usage: NeRF
```sh
PyTorch-InstantNGP> main.py --config_path nerf_config.yaml
```
Download Lego Nerf dataset from [here](https://drive.google.com/drive/folders/1lrDkQanWtTznf48FCaW5lX9ToRdNDF1a)

### Usage: Image
```sh
PyTorch-InstantNGP> main.py --config_path image_config.yaml
```

All settings in config yaml file.
