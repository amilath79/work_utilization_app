@echo off
echo Training all punch codes with new pipeline...

python train_models3.py --punch-code 202
python train_models3.py --punch-code 203
python train_models3.py --punch-code 206
python train_models3.py --punch-code 209
python train_models3.py --punch-code 210
python train_models3.py --punch-code 211
python train_models3.py --punch-code 213
python train_models3.py --punch-code 214
python train_models3.py --punch-code 215
python train_models3.py --punch-code 217

echo All models trained!
pause