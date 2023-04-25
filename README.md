# QR-Code pattern recognition

How to run the recognizer:
```
python3 main.py --image-path ./images/TestSet1/0001.jpg --result-path ./output/result.jpg --algorithm own
```

There are two available algorithms:
- `own` - Own implementation of qr code recognition
- `cv2` - Cv2 implementation

How to run benchmark
```
python3 compare.py --images-path images_dir
```

Benchmark results:
- TestSet1: `Precision: 0.98901, Recall: 0.96774, Overall time: 29 mins`
- TestSet2: `Precision: 0.98863, Recall: 0.96666, Overall time: 35 mins`
- Average recognition time (per image): `± 30 secs` (yeah, python is slow)
