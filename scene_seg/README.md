## Install Dependencies
The lines below should set up a fresh environment with everything you need: 
```
conda create --name bev
source activate bev 
conda install pytorch=1.12.0 torchvision=0.13.0 cudatoolkit=11.3 -c pytorch
conda install pip
pip install -r requirements.txt
```


## Training
To train MCE model, after activate `bev`, run a command like this:

```
sh train_imml_mce.sh
```


## Code Notes (provided by SimpleBEV)
##### Tensor shapes

We maintain consistent axis ordering across all tensors. In general, the ordering is `B,S,C,Z,Y,X`, where

- `B`: batch
- `S`: sequence (for temporal or multiview data)
- `C`: channels
- `Z`: depth
- `Y`: height
- `X`: width

This ordering stands even if a tensor is missing some dims. For example, plain images are `B,C,Y,X` (as is the pytorch standard).

##### Axis directions

- Z: forward
- Y: down
- X: right

This means the top-left of an image is "0,0", and coordinates increase as you travel right and down. `Z` increases forward because it's the depth axis.

##### Geometry conventions

We write pointclouds/tensors and transformations as follows:

- `p_a` is a point named `p` living in `a` coordinates.
- `a_T_b` is a transformation that takes points from coordinate system `b` to coordinate system `a`.

For example, `p_a = a_T_b * p_b`.

This convention lets us easily keep track of valid transformations, such as
`point_a = a_T_b * b_T_c * c_T_d * point_d`.

For example, an intrinsics matrix is `pix_T_cam`. An extrinsics matrix is `cam_T_world`. 

In this project's context, we often need something like this:
`xyz_cam0 = cam0_T_cam1 * cam1_T_velodyne * xyz_velodyne`
