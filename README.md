This project is for exploring the Echo State Network architecture of Reservoir Computing

[timeSeriesForecast.py](timeSeriesForecast.py) is a shallow ESN that utilizes the TensorFlow Addons ESN architecture and the Jena Climate dataset for predicting the forecast 

[mixedSelectivityTask.py](mixedSelectivityTask.py) compares an ESN with a standard ANN on a visual task similar to the task in the [original mixed selectivity paper](https://www.nature.com/articles/nature12160). Even though the ESN had less trainable parameters, it had significantly higher accuracy on the visual task than the ANN (0.75 vs 0.32). This suggests that projecting inputs into nonlinear subspaces can be efficient for training in conditions with limited trainable parameters. 

[mixedSelectivityComparison.py](mixedSelectivityComparison.py) is an analysis of the hyperparameters optimal for this task and a look into the effectiveness of reservoir computing for tasks in which trainable layer sizes are limited. It uses the same models as mixedSelectivityTask but performs the test on 5 different Layer 2 sizes, ranging from 10-50, and 5 different reservoir layer sizes, ranging from 100-500. In each case, the classic model has over 5x the number of trainable parameters of the RC, but the classic model lacks the high dimensional function availability of the reservoir layer. Keep in mind this is a difficult visual task, so 80% is quite high given the limited number of trainable layers. Here are the results:

| L2 Size    | Classic Model | Reservoir size 100 | Reservoir size 200 | Reservoir size 300 | Reservoir size 400 | Reservoir size 500 |
| -------- | ------- | ------- |------- |------- |------- |------- |
| 10  | 0.3307 | 0.4367 | 0.5570 | 0.5347 | 0.6354 | 0.6801 |
| 20 | 0.2012 |0.4341 | 0.6122 | 0.6059 |  0.7224 | 0.7427 |
| 30   |0.2974| 0.4939| 0.6071 | 0.6833 | 0.7078 | 0.7258 |
| 40   | 0.4754 | 0.4831 |  0.6177 | 0.6733 | 0.7402 | 0.8041 |
| 50   |0.3382 | 0.5781 | 0.6145 | 0.6900 | 0.7410 | 0.8053 |

[mixedSelectivityFewShot.py](mixedSelectivityFewShot.py) is an example of an ESN outperforming a standard model in conditions of low amounts of training data (3000 examples) and limited trainable parameters, with Layer 2 sizes being 50. In these cases with limited trainable parameters and limited layers, the ESN architecture shows its strength. The ESN had an accuracy of 0.7672 while the ANN had an accuracy of  0.5800 in conditions of low training data. 
