This project is for exploring the Echo State Network architecture of Reservoir Computing

[timeSeriesForecast.py](timeSeriesForecast.py) is a shallow ESN that utilizes the TensorFlow Addons ESN architecture and the Jena Climate dataset for predicting the forecast 

[mixedSelectivityTask.py](mixedSelectivityTask.py) compares an ESN with a standard ANN on a visual task similar to the task in the [original mixed selectivity paper](https://www.nature.com/articles/nature12160). Even though the ESN had less trainable parameters, it had significantly higher accuracy on the visual task than the ANN (0.75 vs 0.32). This suggests that projecting inputs into nonlinear subspaces can be efficient for training in conditions with limited trainable parameters. 


