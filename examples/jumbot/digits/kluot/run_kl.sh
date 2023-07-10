python train.py --source_dset mnist --target_dset usps > "m2u.txt"

python train.py --source_dset svhn --target_dset usps > "s2u.txt"

python train.py --source_dset usps --target_dset mnist > "u2m.txt"

python train.py --source_dset mmnist --target_dset usps > "mm2u.txt"

python train.py --source_dset mmnist --target_dset mnist > "mm2m.txt"

python train.py --source_dset svhn --target_dset mmnist > "s2mm.txt"

python train.py --source_dset mnist --target_dset mmnist > "m2mm.txt"

python train.py --source_dset svhn --target_dset mnist > "s2m.txt"
