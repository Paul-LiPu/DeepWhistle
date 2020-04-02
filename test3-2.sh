#python 3.Network_train_and_test/train.py --data_type h5 \
#  --train_data 3.Network_train_and_test/train_DWC-I.txt \
#  --test_data 3.Network_train_and_test/test.txt \
#  --exp_name DWC-I

#python 3.Network_train_and_test/train.py --data_type lmdb \
#  --train_data 3.Network_train_and_test/train_DWC-II.txt \
#  --test_data 3.Network_train_and_test/test.txt \
#  --exp_name DWC-II

#python 3.Network_train_and_test/train.py --data_type lmdb \
#  --train_data 3.Network_train_and_test/train_DWC-III.txt \
#  --test_data 3.Network_train_and_test/test.txt \
#  --exp_name DWC-III

python 3.Network_train_and_test/train.py --data_type lmdb \
  --train_data 3.Network_train_and_test/train_DWC-IV.txt \
  --test_data 3.Network_train_and_test/test.txt \
  --recall_guided True \
  --recall_val_data 3.Network_train_and_test/val.txt  \
  --pretrained_model 3.Network_train_and_test/models/DWC-III.pth \
  --exp_name DWC-IV



