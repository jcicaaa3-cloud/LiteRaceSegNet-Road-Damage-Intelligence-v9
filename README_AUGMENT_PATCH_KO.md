# LRS augmentation patch

큰 통합 ZIP 대신 데이터셋 part 7개 + 이 작은 패치를 사용합니다.

```bash
cd /home/ubuntu/road-damage-segmentation-portfolio

for z in ~/LRS_DATASET_POTHOLE_BINARY_PART_*_OF_*.zip; do
  unzip -o "$z" -d .
done

unzip -o ~/LRS_AUGMENT_AND_TRAIN_PATCH_SMALL.zip -d .

bash scripts/prepare_aug_dataset.sh

python seg/train_literace.py --config seg/config/pothole_binary_literace_v4_A_aug_highres_damage_boost.yaml
```

GPU 메모리 부족:

```bash
python seg/train_literace.py --config seg/config/pothole_binary_literace_v4_A_aug_safe_damage_boost.yaml
```

recall 더 밀기:

```bash
python seg/train_literace.py --config seg/config/pothole_binary_literace_v4_B_aug_recall_aggressive.yaml
```
