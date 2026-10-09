# Face-recognition data

Place the course image sequences and supplementary files here:

| Input | Location |
| --- | --- |
| Training sequences | `train_data/<person>/%04d.jpg` |
| Test sequences | `test_data/<person>/%04d.jpg` |
| Embedding model | `resnet50_128.onnx` |
| DIR training data | `evaluation_train_data.pkl` |
| DIR test data | `evaluation_test_data.pkl` |
| Individual open-set learning data | `challenge_train_data.csv` |

Training creates `recognition_gallery.pkl` and `clustering_gallery.pkl` in this folder.

Input data and generated caches are excluded from Git.
