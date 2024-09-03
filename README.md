# Deep Learning Module: Charity Funding Predictor Analysis
## Overview
**Purpose**
The goal of this analysis is to provide a nonprofit foundation, Alphabet Soup, a tool that can aid in selecting applicants with the greatest chance of success in their ventures. The analysis was executed utilizing machine learning and neural networks, creating a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features provided in the dataset. 

**Dataset**
Consisting of more than 34,000 organizations that have received funding from Alphabet Soup over the years, the dataset used to preform the analysis includes columns that capture metadata about each organization, including:
-   **EIN**—Identification column
-   **NAME**—Identification column
-   **APPLICATION_TYPE**—Alphabet Soup application type
-   **AFFILIATION**—Affiliated sector of industry
-   **CLASSIFICATION**—Government organization classification
-   **USE_CASE**—Use case for funding
-   **ORGANIZATION**—Organization type
-   **STATUS**—Active status
-   **INCOME_AMT**—Income classification
-   **SPECIAL_CONSIDERATIONS**—Special considerations for application
-   **ASK_AMT**—Funding amount requested
-   **IS_SUCCESSFUL**—Was the money used effectively

## Data & Modeling Approach
### STEP 1: Data Preprocessing
1. Read in the `charity_data.csv` to a Pandas DataFrame, and be sure to identify the following in your dataset:
* **Target Variables of this Model**
	 * **IS_SUCCESSFUL**—Was the money used effectively
		* 0 if not successful
		* 1 if successful
* **Feature Variables of this Model**
	* **APPLICATION_TYPE**—Alphabet Soup application type
	* **AFFILIATION**—Affiliated sector of industry
	* **CLASSIFICATION**—Government organization classification
	* **USE_CASE**—Use case for funding
	* **ORGANIZATION**—Organization type
	* **STATUS**—Active status
	* **INCOME_AMT**—Income classification
	* **SPECIAL_CONSIDERATIONS**—Special considerations for application
	* **ASK_AMT**—Funding amount requested
* **Variables Removed from the Input Data of this Model**
	* **EIN**—Identification column
	* **NAME**—Identification column
2.  Determine the number of unique values for each column. For columns that have more than 10 unique values, determine the number of data points for each unique value.
3.  Use the number of data points for each unique value to pick a cutoff point to combine "rare" categorical variables together in a new value,  `Other`, and then check if the replacement was successful.
4.  Use  `pd.get_dummies()`  to encode categorical variables.
5.  Split the preprocessed data into a features array,  `X`, and a target array,  `y`. Use these arrays and the  `train_test_split`  function to split the data into training and testing datasets.
6.  Scale the training and testing features datasets by creating a  `StandardScaler`  instance, fitting it to the training data, then using the  `transform`  function.

### STEP 2: Compiling, Training, and Evaluating the Model
1. **Design Neural Network**
* Create a neural network model: assign the number of input features, nodes/neurons for each layer, create hidden layers, create output layer, assign appropriate activation functions to each layer, using TensorFlow and Keras. 
	* **Neurons, Layers, and Activation Functions***
		* **1st Hidden Layer:*** Neurons = 80, Dense, Activation Function = *"relu"*
		* **2nd Hidden Layer:*** Neurons = 30, Dense, Activation Function = *"relu"*
		* **Output Layer:*** Neurons = 1, Dense, Activation Function = *"sigmoid"*
* Check the structure of the model using `nn.summary()`
2. **Compile the Model:**
```nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])``` 
3. **Train the Model:** 
```fit_model = nn.fit(x_train_scaled, y_train, epochs=100)```
The number of epochs chosen was 100, as it is neural network standard used when working with smaller data sets. If further performance improvement is desired, this number may be increased incrementally, however the risk of overfitting the model increases as well.
5. **Evaluate the model** using `nn.evaluate()` and the test data to determine the loss and accuracy.

### STEP 3: Optimizing the Model
*Optimize model to achieve a target predictive accuracy higher than 75%*
1. **Model Optimization Techniques**
	* Dropping more columns - **STATUS** and **SPECIAL_CONSIDERATIONS** were dropped due to small value counts, in addition to **EIN**. **NAME** was not dropped during the optimization process.
	* Increasing / decreasing the number of values for each bin - **NAME** counts less than or equal to 5 were moved into a bin named `Other`.
	* Add more neurons to a hidden layer - 
		* **1st Hidden Layer:*** Neurons = 100
		* Increasing the number of neurons from 80 to 100 in the first hidden layer increases data availability in the subsequent layers, enabling improved speeds and loss reduction.
	* Add more hidden layers - 
		* **3rd Hidden Layer:*** Neurons = 10
		* The addition of the 3rd hidden layer allows for more complex network interaction with the input variables.
	* Use different activation functions for the hidden layers -
		* **2nd Hidden Layer:*** Activation Function = *changed from "relu" to "sigmoid"*
		* Increasing the complexity of activation functions as the layers progress allow for more diverse assessment of the more complex features of the input data without impacting the assessment of the less complex features within the same input data. The sigmoid function is ideal for this binary classification model.   
		
## Results
**Initial Model**
* **Loss =** 0.5590384602546692
* **Accuracy =** 0.7365597486495972
* **Random forest model accuracy =** 71.21
		
**Optimized Model**
* **Loss =** 0.46084949374198914
* **Accuracy =** 0.7927696704864502
* **Random forest model accuracy =** 77.63
## Summary

The initial model provided sufficient, valuable data output with a loss of approximately 0.56 (56%) and accuracy of 0.74 (74%). This illustrates that errors occurred in more than half of the iterations performed by this model. It can also be stated that a majority of predictions made by this model were correct.

The optimized model provided a slight improvement in performance with a loss of approximately 0.46 (46%), and accuracy of 0.79 (79%). This illustrates that by adjusting the hyperparameters, the percentage of correct predictions increased by 5% and the model's iteration errors decreased by 10%.

After reviewing the results, it can be said that the binary classification model does an acceptable job at predicting which applicants seeking funding from Alphabet Soup possess the greatest chance of success. 

The binary classification model used in this analysis is acceptable, however alternative models such as Support Vector Machine (SVMs) or Random Forests, could yield improved performance scores.   
* The SVMs are effective when applied to smaller datasets and binary classification problems. Unlike neural networks, SVMs consistently drawn margins mid-way between the closest points of the two classes, regardless of initialization of the weights or the order the training sets are presented. SVMs generally provide higher accuracy scores and faster prediction performances, in comparison to the model used in this analysis (F., 2021). 
* Random Forests require much less data preprocessing and perform well with binary features, categorical features, and numerical features, eliminating the need for feature normalization. Random Forests advantages include versatile use, easy-to-understand hyperparameters, and reduction in the risk of overfitting, assuming there are enough trees in the forest (Donges, n.d.).

## References

Donges, N. (n.d.). _A complete guide to the random forest algorithm_ (B. Whitfield & S. Pierre, Eds.). Built In. Retrieved September 2, 2024, from https://builtin.com/data-science/random-forest-algorithm#:~:text=Random%20forest%20is%20a%20flexible

F., I. (2021, February 18). _Are neural networks better than SVMs?_ StackExchange; Cross Validated. https://stats.stackexchange.com/a/510100


> Written with [StackEdit](https://stackedit.io/).
