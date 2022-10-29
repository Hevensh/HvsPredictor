# HvsPredictor
use a specified loss to forecast the single column data
test by Hevensh

MyPredictor.py store the class used in the experiment following, which contain some process to pre-process the data and some evaluation methods. 
The train method about is use a special loss function to find some feature include the predictability od data, and predict the value next time point if it is predictable or return value near null if the model feel it is difficult to predict. this is all about Mypredict.  

Track_remake.ipynb is use to demonstrate the difference of the performance between the proposed loss function and MSE on the almost impredictable data.

predictable_test.ipynb is use to demonstrate the performance on a predictable generated data.

Good luck and have fun. / >w< /
