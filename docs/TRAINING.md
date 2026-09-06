# Detector training

Use the deployer's full installation before opening the Train tab.

1. Choose a save directory. The application creates `datapoints/` there and
   writes the trained model as `model.h5`.
2. Keep the default SAM2 checkpoint or choose another compatible `.pt` file.
3. Collect valid examples by selecting flake pixels in a disposable image set.
4. Collect invalid examples by selecting SAM2 regions and sampling their grid.
5. Run label and train.
6. Save a copy of the resulting model through the existing Save Model action.

Defaults remain unchanged: 1024 maximum display width, 128 SAM2 grid samples,
100 epochs, batch size 32, test split 0.20, and patience 10.

Training and SAM2 modules are imported only when their operations are invoked.
Opening the application or using stage/capture controls does not load
TensorFlow, PyTorch, scikit-learn, or SAM2. TensorFlow is loaded when a detector
model is selected or model training begins; PyTorch/SAM2 load only for invalid
data collection.

Training outputs and collected data remain in the user-selected directory. The
deployer and repository cleanup never prune them.
