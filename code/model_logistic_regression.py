# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import yfinancetool as yft
import techinal_indicato as ti
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import auc, precision_score, recall_score, roc_curve
from keras.utils import to_categorical

class LogisticRegressionHelper(): 
      

    def train_logistic_regression_model(self, df_train,features, target):
        
        # Create features and target
        df_features = df_train[features]
        target = df_train[target]  # Using SMA Signal as target for this example

        # # Split the data
        X_train, X_test, y_train, y_test = train_test_split(df_features, target, test_size=0.3, random_state=42)

        # Apply Chi-Squared feature selection
        chi2_selector = SelectKBest(chi2, k='all')
        chi2_selector.fit(X_train, y_train)

        # Get the selected features
        selected_features = chi2_selector.get_support(indices=True)
        print(f'Selected features (CHI): {selected_features}')
        print(f'Selected features (CHI): {df_features.columns[selected_features]}')
        
        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # # Train the logistic regression model
        # model = LogisticRegression(max_iter=100)
        # model.fit(X_train_scaled, y_train)

        # # Make predictions
        # y_pred = model.predict(X_test_scaled)

        # # Evaluate the model
        # print(confusion_matrix(y_test, y_pred))
        # print(classification_report(y_test, y_pred))

        clf = [
        LogisticRegression(solver='newton-cg',penalty='l2',max_iter=500),
        LogisticRegression(solver='lbfgs',penalty='l2',max_iter=500),
        LogisticRegression(solver='sag',penalty='l2',max_iter=500),
        LogisticRegression(solver='saga',penalty='l2',max_iter=500),
        LogisticRegression(solver='liblinear',penalty='l1',max_iter=500)
        ]
        clf_columns = []
        clf_compare = pd.DataFrame(columns = clf_columns)

        row_index = 0
        for alg in clf:
                
            predicted = alg.fit(X_train, y_train).predict(X_test)
            fp, tp, th = roc_curve(y_test, predicted)
            clf_name = alg.__class__.__name__
            clf_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 5)
            clf_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 5)
            clf_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted),5)
            clf_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted),5)
            clf_compare.loc[row_index, 'AUC'] = round(auc(fp, tp),5)

            row_index+=1
            
        clf_compare.sort_values(by = ['Test Accuracy'], ascending = False, inplace = True)    
        print(clf_compare)

        return clf[0], scaler

class GradientBoostClassifierHelper(): 

    def train_gradient_classifier_model(self, df_train,features, target):
        #    https://stackoverflow.com/questions/56505564/handling-unbalanced-data-in-gradientboostingclassifier-using-weighted-class
        # Create features and target
        df_features = df_train[features]
        target = df_train[target]  # Using SMA Signal as target for this example
        

        # # Split the data
        X_train, X_test, y_train, y_test = train_test_split(df_features, target, test_size=0.3, random_state=42)
        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        sample_weights = np.zeros(len(y_train))
        sample_weights[y_train == 0] = 0.2
        sample_weights[y_train == 1] = 0.4
        sample_weights[y_train == -1] = 0.4

        #Define the parameter grid
        #'learning_rate': 0.2, 'max_depth': 6, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 300, 'subsample': 0.8}
        param_grid = {
            'n_estimators': [200,500,1000], #,[ 1000,2000], #200,500,1000
            'learning_rate': [0.2], #[0.2], #[0.01, 0.1, 0.2]
            'max_depth': [7, 8, 9] #3, 4, 5, [5, 6, 7]    
                }
        
    # 'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 500
        # param_grid = {
        #     'n_estimators': [100, 200, 300],
        #     'learning_rate': [0.05, 0.1, 0.2],
        #     'max_depth': [3, 4, 5, 6],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 4],
        #     'subsample': [0.8, 0.9, 1.0]
        # }
        # Initialize the Gradient Boosting Regressor
        gbm = GradientBoostingClassifier(random_state=42)

        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid)

        class_weight = y_train.value_counts(normalize=True).to_dict()
        sample_weight = y_train.map(lambda x: 1/class_weight[x])
        # Fit GridSearchCV
        grid_search.fit(X_train_scaled, y_train, sample_weight = sample_weight)

        # Get the best parameters
        best_params = grid_search.best_params_
        print(f'Best parameters: {best_params}')

        # Train the model with the best parameters
        best_gbm = grid_search.best_estimator_

        # Make predictions
        y_pred = best_gbm.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error after tuning: {mse}')
        return best_gbm, scaler

class LongShortTermMemoryMLHelper(): 
    # Function to create a dataset for LSTM
    def create_dataset(self, data, labels, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step)])
            y.append(labels[i + time_step])#time_step - 1
        return np.array(X), np.array(y)
    
    def train_lstm_model(self, df_train,features, target, time_step=28):
        df_features = df_train[features]
        df_target = df_train[target]  # Using SMA Signal as target for this example
        # One hot encode the labels
        one_hot_labels = to_categorical(df_target,num_classes = 3)
        #X_train, X_test, y_train, y_test = train_test_split(df_features, target, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(df_features)
        X, y = self.create_dataset(X_train_scaled, one_hot_labels, time_step)
        print(X.shape[2],(time_step, X.shape[2]))
        print(X.shape)
        print(y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        print(X_train.shape)
        print(X_test.shape)
        # Step 3: Build the LSTM Model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X.shape[2])))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(3, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Step 4: Train the Model
        model.fit(X_train, y_train, epochs=100, batch_size=1024, validation_data=(X_test, y_test))

        # Step 5: Evaluate and Predict
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        # Plot actual vs predicted signals

        train_size = len(X_train) 
        plt.figure(figsize=(14, 7))
        plt.plot(df_train.index[train_size + time_step:], y_test_classes, label='Actual Signal')
        plt.plot(df_train.index[train_size + time_step:], y_pred_classes, label='Predicted Signal', alpha=0.7)
        plt.legend()
        plt.show()
        return model, scaler
        #return 1, 2

