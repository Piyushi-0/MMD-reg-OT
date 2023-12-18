## Code for our work [MMD-Regularized Unbalanced Optimal Transport](https://arxiv.org/pdf/2011.05001.pdf)
- You may first install PyTorch and Torchvision via `conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia`.
- The other packages used in this repository can be installed via `pip install -r requirements.txt`.
- To install our `ot_mmd` package, please clone this repository and run `pip install .`

### Algorithms
  - [Code for solving MMD-OT using Accelerated PGD.](./ot_mmd/mmdot.py)
      - [Code for computing a batch of MMD-OT problems parallelly.](./ot_mmd/b_mmdot.py)
  - [Code for solving the MMD-OT barycenter problem using Accelerated PGD.](./ot_mmd/barycenter.py)
### Basic Examples
  - [OT plan between Gaussians.](./examples/synthetic/OTplan.ipynb)
  - [Barycenter between Gaussians.](./examples/synthetic/barycenter_with_imq.ipynb)
### [More Examples](./examples)


**If you find the code useful, consider giving a star to this repository & [citing our work](https://github.com/Piyushi-0/MMD-reg-OT/blob/main/bibtex.txt).**


