import time
import numpy as np
from tensorflow import keras

# --- User settings ---
model_path    = "/Users/massimozhang/Desktop/coding/Auto_Scan1/TIT_10x.h5"
image_path    = "/Users/massimozhang/Desktop/coding/Auto_Scan1/data/TIT/10x/training_data/1_10x.png"
ratio         = 14
feature_dim   = 6  # adjust if you compute more features per grid point
rows, cols    = None, None  # will be inferred from the image
coords, feats = None, None

# load model
model = keras.models.load_model(model_path)

# prepare a single grid-features array once
import cv2
img_bgr = cv2.imread(image_path)
h, w    = img_bgr.shape[:2]
rows, cols = int(h/ratio), int(w/ratio)

# dummy features (you would compute these as in your real pipeline)
# here we just make random data of shape (rows*cols, feature_dim)
N = rows * cols
X = np.random.rand(N, feature_dim).astype(np.float32)

def benchmark(batch_sizes, repeats=3):
    results = {}
    for bs in batch_sizes:
        # run a few repeats and take the best time
        times = []
        for _ in range(repeats):
            start = time.time()
            _ = model.predict(X, batch_size=bs, verbose=0)
            times.append(time.time() - start)
        best = min(times)
        samples_per_sec = N / best
        results[bs] = samples_per_sec
        print(f"batch_size={bs:4d} → {samples_per_sec:8.1f} samples/sec (best of {repeats})")
    return results

if __name__ == "__main__":
    #candidates is a list of the batch size to try, input yours and see what is best
    candidates = [10000*i for i in range(15,26)]  # 32,64,...,4096
    bench = benchmark(candidates)
    best_bs = max(bench, key=bench.get)
    print(f"\nOptimal batch size on this hardware: {best_bs}")
