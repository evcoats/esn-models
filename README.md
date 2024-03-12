This project is for exploring the Echo State Network architecture of Reservoir Computing

[timeSeriesForecast.py](timeSeriesForecast.py) is a shallow ESN that utilizes the TensorFlow Addons ESN architecture and the Jena Climate dataset for predicting the forecast 

[mixedSelectivityTask.py](mixedSelectivityTask.py) compares an ESN with a standard ANN on a visual task similar to the task in the [original mixed selectivity paper](https://www.nature.com/articles/nature12160). Even though the ESN had less trainable parameters, it had significantly higher accuracy on the visual task than the ANN (0.75 vs 0.32). This suggests that projecting inputs into nonlinear subspaces can be efficient for training in conditions with limited trainable parameters. 

mixedSelectivityComparison is an analysis of the hyperparameters optimal for this task and a look into the effectiveness of reservoir computing for tasks in which trainable layer sizes are limited. It uses the same models as mixedSelectivityTask but performs the test on 5 different L2 sizes, ranging from 10-50, and 5 different reservoir layer sizes, ranging from 100-500. In each case, the classic model has over 5x the number of trainable parameters of the RC, but the classic model lacks the high dimensional function availability of the reservoir layer. Here are the results:

| L2 Size    | Classic Model | Reservoir size 100 | Reservoir size 200 | Reservoir size 300 | Reservoir size 400 | Reservoir size 500 |
| -------- | ------- | ------- |------- |------- |------- |------- |
| 10  |     |
| 20 |      |
| 30   | $420    |
| 40   | $420    |
| 50   | $420    |
