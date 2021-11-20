wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/all_csv_files.csv
mv all_csv_files.csv /work/shared/common/project_build/gnn-optane/data/all_csv_files.csv
python amazon-review.py
cd ../
mkdir amazon-review
mv load_amazon-review/user_item_list.txt amazon-review/train.txt