python mmd.py --t_pred 1 --save_as results
python kluot.py --t_pred 1 --best_lda 10 --best_hp 0.01 --save_as results
python proposed.py --t_pred 1 --best_lda 1 --best_hp 0.1 --save_as results

python mmd.py --t_pred 2 --save_as results
python kluot.py --t_pred 2 --best_lda 1 --best_hp 0.1 --save_as results
python proposed.py --t_pred 2 --best_lda 1 --save_as results

python mmd.py --t_pred 3 --save_as results
python kluot.py --t_pred 3 --best_lda 1 --best_hp 0.1 --save_as results
python proposed.py --t_pred 3 --best_lda 1 --save_as results
