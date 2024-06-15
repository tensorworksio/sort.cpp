# SORT

Yet another C++ adaptation of SORT tracking algorithm

## Dataset

```bash
git clone https://github.com/tensorworksio/sort.cpp
cd sort.cpp
wget https://motchallenge.net/data/MOT15.zip
unzip MOT15.zip -d data && rm MOT15.zip
```

## Dependencies

- boost
- dlib
- opencv
- nlohmann/json

## Compile

```bash
meson setup build
cd build
ninja
```
## Configure
Set your SORT configuration in `config.json`

## Run
Launch the MOT Challenge

```bash
cd build
./main --path ../data/MOT15/train/ADL-Rundle-8 --config ../config.json --display
```
Or if you want to enable perfect object detector

```bash
./main --path ../data/MOT15/train/ADL-Rundle-8 --config ../config.json --gt --display
```

## Test
```bash
python3 -m pip install pipenv
python3 -m pipenv install
# Run evaluation
python3 -m pipenv run python mot-eval.py --path=data/MOT15/train/ADL-Rundle-8
```
