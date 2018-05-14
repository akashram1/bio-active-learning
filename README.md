# bio-active-learning
Application of Active Learning in classifcation of images of cell organelles

## Background
This project uses a simple multi-layer perceptron (MLP) to classifies sub-cellular images of various organelles in an Active Learning setup.
As labels of such images are expensive to get, we must identify only those images which help the classifier (MLP) learn the most / come closest to the optimal parameter space. 
By setting a budget on the maximum number of labels of images to obtain from an 'oracle', we can train an MLP with only a 'budget' number of images. We select the best 
'budget' number of images using 3 methods:
 - Best-Versus-Second-Best (BvSB) approach : The data point that is selected in every iteration of active learning is the one which has the smallest
difference between the top-2 class-probabilities calculated for it by the base learner. Greater this difference, smaller the uncertainty of the point.
 - Entropy approach : In every active learning iteration, the point which has the highest entropy will be added to the pool of labeled instances L that will be used to update the hypothesis space (in this project, the weights and biases of the neural network).


