# SORT
<p align="center">
    <img src="https://github.com/tensorworksio/sort.cpp/blob/master/docs/output.gif" width="640" height="360"/>
</p>

Yet another C++ adaptation of SORT multi object tracking algorithm, inspired by [motracker](https://github.com/adipandas/multi-object-tracker/tree/master)

## Dataset

```bash
git clone https://github.com/tensorworksio/sort.cpp
cd sort.cpp
wget https://motchallenge.net/data/MOT15.zip
unzip MOT15.zip -d data && rm MOT15.zip
```

## Compile

```bash
meson setup build
meson compile -C build
```
## Configure
Set your SORT configuration in `config.json`

## Run
Launch the MOT Challenge

```bash
cd build
./sort --help
```

### Example

```bash
cd build
./sort --path ../data/MOT15/train/ADL-Rundle-8 --config ../config.json --display --gt --save
```

## Test
```bash
python3 -m venv venv
. venv/bin/activate
pip3 install -r requirements

python3 tests/mot-eval.py --path data/MOT15/train/ADL-Rundle-8
```
