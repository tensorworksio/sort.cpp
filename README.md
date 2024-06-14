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

## Compile

```bash
meson setup build
cd build
ninja
```

## Run

```bash
cd build
./main --path=../data/MOT15/train/ADL-Rundle-8 --display
```

## Test

```bash
python3 -m pip install pipenv
python3 -m pipenv install
# Run evaluation
python3 -m pipenv run python mot-eval.py --path=data/MOT15/train/ADL-Rundle-8
```
