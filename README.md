# Semi-supervised Adversarial Learning for Complementary Item Recommendation

Dataset is taken from:
http://deepyeti.ucsd.edu/jianmo/amazon/index.html

Download the metadata file and place it under ./data/raw
```
├── LICENSE
├── README.md
├── configs
├── data
    ├── final
    ├── processed
    └── raw
        ├── meta_Clothing_Shoes_and_Jewelry.json.gz
        └── meta_Toys_and_Games.json.gz
├── notebooks
├── outputs
├── requirements.txt
└── src
```

## Scripts
```
python main_process_meta.py category_name=Clothing_Shoes_and_Jewelry
python main_download_imgs.py category_name=Clothing_Shoes_and_Jewelry
python main_create_embeddings.py category_name=Clothing_Shoes_and_Jewelry
python main_create_train_test_set.py category_name=Clothing_Shoes_and_Jewelry
python main_execute_train.py category_name=Clothing_Shoes_and_Jewelry note="_triplet" discriminator_weight=0.0 clf_weight=0.0 cycle_weight=0.0
python main_execute_train.py category_name=Clothing_Shoes_and_Jewelry note="_triplet_and_cycle_" discriminator_weight=0.0 clf_weight=0.0 
python main_execute_train.py category_name=Clothing_Shoes_and_Jewelry note="_clf_and_cycle_" triplet_weight=0.0 discriminator_weight=0.0
python main_execute_train.py category_name=Clothing_Shoes_and_Jewelry
python main_execute_train.py category_name=Clothing_Shoes_and_Jewelry method="dcf"
python main_execute_train.py category_name=Clothing_Shoes_and_Jewelry method="dcf" hard_negative=false

```

```
python main_process_meta.py category_name=Toys_and_Games
python main_download_imgs.py category_name=Toys_and_Games
python main_create_embeddings.py category_name=Toys_and_Games
python main_create_train_test_set.py category_name=Toys_and_Games
python main_execute_train.py category_name=Toys_and_Games note="_triplet" discriminator_weight=0.0 clf_weight=0.0 cycle_weight=0.0 
python main_execute_train.py category_name=Toys_and_Games note="_triplet" discriminator_weight=0.0 clf_weight=0.0 cycle_weight=0.0
python main_execute_train.py category_name=Toys_and_Games note="_triplet_and_cycle_" discriminator_weight=0.0 clf_weight=0.0 
python main_execute_train.py category_name=Toys_and_Games note="_clf_and_cycle_" triplet_weight=0.0 discriminator_weight=0.0
python main_execute_train.py category_name=Toys_and_Games
python main_execute_train.py category_name=Toys_and_Games method="dcf"
python main_execute_train.py category_name=Toys_and_Games method="dcf" hard_negative=false
```

```
python main_process_meta.py category_name=Home_and_Kitchen
python main_download_imgs.py category_name=Home_and_Kitchen
python main_create_embeddings.py category_name=Home_and_Kitchen
python main_create_train_test_set.py category_name=Home_and_Kitchen
python main_execute_train.py category_name=Home_and_Kitchen note="_triplet" discriminator_weight=0.0 clf_weight=0.0 cycle_weight=0.0
python main_execute_train.py category_name=Home_and_Kitchen note="_triplet_and_cycle_" discriminator_weight=0.0 clf_weight=0.0 
python main_execute_train.py category_name=Home_and_Kitchen note="_clf_and_cycle_" triplet_weight=0.0 discriminator_weight=0.0
python main_execute_train.py category_name=Home_and_Kitchen
python main_execute_train.py category_name=Home_and_Kitchen method="dcf"
python main_execute_train.py category_name=Home_and_Kitchen method="dcf" hard_negative=false
python main_execute_train.py category_name=Home_and_Kitchen method="pcomp" 
```

python main_execute_train.py category_name=Clothing_Shoes_and_Jewelry method="pcomp" note="_addone" \;
python main_execute_train.py category_name=Toys_and_Games method="pcomp" note="_addone" \;
python main_execute_train.py category_name=Home_and_Kitchen method="pcomp" note="_addone"
