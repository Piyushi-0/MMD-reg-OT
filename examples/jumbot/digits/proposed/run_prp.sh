python train.py --lda 100 --source_dset usps --target_dset mnist --khp 1e-2 --ktype imq_v2 --case bal --crit rgrad > "new_u2m.txt"
python train.py --lda 100 --source_dset mmnist --target_dset usps --khp 1e-2 --ktype imq_v2 --case bal --crit rgrad > "new_mm2u.txt"
python train.py --lda 100 --source_dset mmnist --target_dset mnist --khp 1e-2 --ktype imq_v2 --case bal --crit rgrad > "new_mm2m.txt"
python train.py --lda 100 --source_dset svhn --target_dset mmnist --khp 1e-2 --ktype imq_v2 --case bal --crit rgrad > "new_s2mm.txt"
python train.py --lda 100 --source_dset mnist --target_dset mmnist --khp 1e-2 --ktype imq_v2 --case bal --crit rgrad > "new_m2mm.txt"
python train.py --lda 100 --source_dset svhn --target_dset mnist --khp 1e-2 --ktype imq_v2 --case bal --crit rgrad > "new_s2m.txt"
python train.py --lda 100 --source_dset mnist --target_dset usps --khp 1e-2 --ktype imq_v2 --case bal --crit rgrad > "new_m2u.txt"
python train.py --lda 100 --source_dset svhn --target_dset usps --khp 1e-2 --ktype imq_v2 --case bal --crit rgrad > "new_s2u.txt"