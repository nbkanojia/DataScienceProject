# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import yfinancetool as yft
import techinal_indicato as ti
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler,MinMaxScaler,label_binarize
from sklearn.metrics import classification_report, confusion_matrix,make_scorer, precision_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization,Input,Flatten, GRU
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import auc, precision_score, recall_score, roc_curve,roc_auc_score,precision_recall_curve,f1_score
from keras.utils import to_categorical
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight,compute_sample_weight
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler,ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.utils import resample
from kerastuner.tuners import RandomSearch
import math
from sklearn.model_selection import TimeSeriesSplit
import kerastuner as kt
import plots
from tensorflow.keras.optimizers import Adam
from scipy.stats import uniform
from sklearn.base import BaseEstimator, ClassifierMixin
import pickle

class LogisticRegressionHelper(): 
      
    def predict_signals(self,model,scaler,df,features):

        df_features = df[features]
        X_test_scaled = scaler.transform(df_features)
        y_pred = model.predict(X_test_scaled)
        return pd.Series(y_pred)
    def predict_signals_from_saved_model(self,df,features):
        with open('scaler_lr.pkl','rb') as f:
            scaler = pickle.load(f)
        with open('best_logistic_model.pkl', 'rb') as file:
            model = pickle.load(file)
        df_features = df[features]
        X_test_scaled = scaler.transform(df_features)
        y_pred = model.predict(X_test_scaled)
        return pd.Series(y_pred)
    def train_logistic_regression_model(self, df_train,features, target):
        
        # Create features and target
        df_features = df_train[features]
        target = df_train[target]  # Using SMA Signal as target for this example

        # # Split the data
        X_train, X_test, y_train, y_test = train_test_split(df_features, target, test_size=0.4, random_state=42)
        print(len(y_test))
        # Standardize the features
        scaler = StandardScaler()
        #scaler.fit_transform(df_features)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        with open('scaler_lr.pkl','wb') as file:
            pickle.dump(scaler,file)

        #Define the parameter grid
        param_grid = {
            'solver': ['newton-cg','lbfgs','sag','saga','liblinear'], #,[ 1000,2000], #200,500,1000
            'C': [0.001, 0.001,0.01,0.1,1], #[0.2], #[0.01, 0.1, 0.2]
        }
      
        
         
        # Initialize the Logistic Regressor
        lr = LogisticRegression(penalty='l2',max_iter=200,class_weight='balanced')

        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=lr, param_grid=param_grid,scoring='f1')
        
        grid_search.fit(X_train_scaled, y_train)#

        # Get the best parameters
        best_params = grid_search.best_params_
        print(f'Best parameters: {best_params}')

        # Train the model with the best parameters
        best_lr = grid_search.best_estimator_

        with open('logistic_model.pkl', 'wb') as file:
            pickle.dump(best_lr, file)

        # Make predictions
        y_pred = best_lr.predict(X_test_scaled)
        
        # Evaluate the model
        print(f'Train Accuracy: {round(best_lr.score(X_train, y_train), 5)}')
        print(f'Test Accuracy: {round(best_lr.score(X_test, y_test), 5)}')
        ps = precision_score(y_test, y_pred, average='weighted')
        print(f'Precission: {round(ps, 5)}')
        rs = recall_score(y_test, y_pred, average='weighted')
        print(f'Recall: {round(rs, 5)}')
        f1s = f1_score(y_test, y_pred, average='weighted')
        print(f'f1: {round(f1s, 5)}')
  
        plt_helper = plots.PlotHelper()
        plt_helper.plot_confusion_matrix(y_test,y_pred)
                
        return best_lr, scaler



class GradientBoostClassifierHelper(): 

    def predict_signals(self,model,scaler,df,features):

        df_features = df[features]
        X_test_scaled = scaler.transform(df_features)
        y_pred = model.predict(X_test_scaled)
        return pd.Series(y_pred)
    def predict_signals_from_saved_model(self,df,features):
        with open('scaler_gb.pkl','rb') as f:
            scaler = pickle.load(f)
        with open('best_gradient_boost_model.pkl', 'rb') as file:
            model = pickle.load(file)
        df_features = df[features]
        X_test_scaled = scaler.transform(df_features)
        y_pred = model.predict(X_test_scaled)
        return pd.Series(y_pred)
    def train_gradient_classifier_model(self, df_train,features, target):
        #    https://stackoverflow.com/questions/56505564/handling-unbalanced-data-in-gradientboostingclassifier-using-weighted-class
        #    https://www.kaggle.com/code/alperengin/custom-scoring-using-scikit-learn-make-scorer
        #    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        # Create features and target
        df_features = df_train[features]
        target = df_train[target]  # Using SMA Signal as target for this example
        

        # # Split the data
        X_train, X_test, y_train, y_test = train_test_split(df_features, target, test_size=0.4, random_state=42)
        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        with open('scaler_gb.pkl','wb') as file:
            pickle.dump(scaler,file)
       

        #Define the parameter grid
        param_grid = {
            'n_estimators': [100,150,200], #,[ 1000,2000], #200,500,1000
            'learning_rate': [0.01,0.2,0.3], #[0.2], #[0.01, 0.1, 0.2]
            'max_depth': [10,15,20], #3, 4, 5, [5, 6, 7]    
           
                }
      
        
         
        # Initialize the Gradient Boosting Regressor
        gbm = GradientBoostingClassifier(random_state=42)

        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, scoring='f1')

        # The GradientBoostingClassifier in scikit-learn does not directly support the class_weight
        #class_weight = y_train.value_counts(normalize=True).to_dict()
        #sample_weight = y_train.map(lambda x: 1/class_weight[x])
        sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
        # Fit GridSearchCV
        grid_search.fit(X_train_scaled, y_train ,sample_weight = sample_weight)#

        # Get the best parameters
        best_params = grid_search.best_params_
        print(f'Best parameters: {best_params}')

        # Train the model with the best parameters
        best_gbm = grid_search.best_estimator_

        with open('gradient_boost_model.pkl', 'wb') as file:
            pickle.dump(best_gbm, file)

        # Make predictions
        y_pred = best_gbm.predict(X_test_scaled)

        # Evaluate the model
         
        print(f'Train Accuracy: {round(best_gbm.score(X_train, y_train), 5)}')
        print(f'Test Accuracy: {round(best_gbm.score(X_test, y_test), 5)}')
        ps = precision_score(y_test, y_pred, average='weighted')
        print(f'Precission: {round(ps, 5)}')
        rs = recall_score(y_test, y_pred, average='weighted')
        print(f'Recall: {round(rs, 5)}')
        f1s = f1_score(y_test, y_pred, average='weighted')
        print(f'f1: {round(f1s, 5)}')

       
        plt_helper = plots.PlotHelper()
        plt_helper.plot_confusion_matrix(y_test,y_pred)
                
        return best_gbm, scaler


class LongShortTermMemoryMLHelper(): 
    def __init__(self): 
        self.dataset1 = []
    # Function to create a dataset for LSTM
    def create_train_dataset(self, data, labels, time_step=1):
        X, y = [], []
       
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step)])
        
            y.append(labels[i + time_step])#time_step - 1
        npx = np.array(X)
        npy = np.array(y)
        return npx, npy
    def create_test_dataset(self, data, time_step=1):
        X = []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step)])
        return np.array(X)
           
    
    def build_model(self, hp):
        model = Sequential()
        print(self.input_shape)
        # Tune the number of LSTM layers
        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(LSTM(
                units=hp.Int(f'units_{i}', min_value=10, max_value=100, step=10),
                return_sequences=True if i < hp.get('num_layers') - 1 else False,
                input_shape=(self.input_shape[1], self.input_shape[2]) if i == 0 else None,
                kernel_regularizer=l2(hp.Float(f'l2_{i}', min_value=1e-5, max_value=1e-2, sampling='LOG'))))
            #model.add(BatchNormalization())
            model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.7, step=0.1)))
        
        model.add(Dense(3, activation='softmax', kernel_regularizer=l2(hp.Float('l2_dense', min_value=1e-5, max_value=1e-2, sampling='LOG'))))

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy','precision','recall','f1_score'])
        
        return model
    def train_lstm_model_random_search(self, df_train,features, target):
        
        df_features = df_train[features]
        target = df_train[target]  
        # Shift target values from [-1, 0, 1] to [0, 1, 2] and then encode
        target_shift = target + 1
        y_encoded = to_categorical(target_shift,num_classes = 3)  
        
        scaler = StandardScaler()
        X_scaled1 = scaler.fit_transform(df_features)

        X_scaled,y_encoded = self.create_train_dataset( X_scaled1, y_encoded, self.time_step)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.4, shuffle=False)
        self.input_shape = X_train.shape

        tuner = RandomSearch(
            self.build_model,
            objective='f1_score',
            max_trials=200,
            executions_per_trial=1,
            directory='hyperparameter_tuning_model_final3',
            project_name='lstm_stock_prediction')
        # Early stopping
        early_stopping = EarlyStopping(monitor='f1_score', patience=10, restore_best_weights=True)

        # Learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(monitor='val_precision', factor=0.5, patience=1)

        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(target_shift), y=target_shift)
        total_class_weight = class_weights.sum()
        class_weights_dict = {i : (class_weights[i]/total_class_weight) for i in range(len(class_weights))}

        
        # Perform the hyperparameter search
        tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test), 
                    callbacks=[early_stopping, lr_scheduler], batch_size=512,class_weight=class_weights_dict)

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"""
The optimal number of LSTM layers is {best_hps.get('num_layers')}.
The optimal number of units in each LSTM layer is {[best_hps.get(f'units_{i}') for i in range(best_hps.get('num_layers'))]}.
The optimal dropout rates are {[best_hps.get(f'dropout_{i}') for i in range(best_hps.get('num_layers'))]}.
The optimal L2 regularization values are {[best_hps.get(f'l2_{i}') for i in range(best_hps.get('num_layers'))]}.
The optimal L2 regularization for the Dense layer is {best_hps.get('l2_dense')}.
""")
       
        # Build the best model
        self.model = tuner.hypermodel.build(best_hps)

        # Train the best model
        history =  self.model.fit(X_train, y_train, epochs=100, batch_size=512, validation_data=(X_test, y_test), 
                            callbacks=[early_stopping, lr_scheduler])

        # Evaluate the model
        # loss, accuracy =  self.model.evaluate(X_test, y_test, verbose=0)
        # print(f"Test Accuracy: {accuracy:.2f}")

        # Save the model
        self.model.save('lstm_stock_prediction_model2.h5')

        # Plotting the training history
        import matplotlib.pyplot as plt

        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='test')
        plt.legend()
        plt.show()
        return self.model, scaler

    def train_lstm_model(self, df_train,features, target):

        df_features = df_train[features]
        target = df_train[target]  
        # Shift target values from [-1, 0, 1] to [0, 1, 2] and then encode
        target_shift = target + 1
        #y_encoded = to_categorical(target_shift,num_classes = 3)  
        y_encoded = []
        # Assign 1 to the appropriate index for each target value
        for i, target in enumerate(target_shift):
            if target == 0:
                y_encoded.append([1., 0., 0.])
            if target == 1:
                y_encoded.append([0., 1., 0.])
            if target == 2:
                y_encoded.append([0., 0., 1.])
        y_encoded = np.array(y_encoded)

        scaler = StandardScaler()
        X_scaled1 = scaler.fit_transform(df_features)
        with open('scaler.pkl','wb') as f:
            pickle.dump(scaler, f)
      
        X_scaled,y_encoded = self.create_train_dataset( X_scaled1, y_encoded, self.time_step)
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
      
        # Build the LSTM model
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, 
                            input_shape=(X_scaled.shape[1], X_scaled.shape[2]),
                             kernel_regularizer=l2(0.001),recurrent_dropout=0.2,dropout=0.2))#, kernel_regularizer=l2(0.001)

        #self.model.add(Dropout(0.5))
        self.model.add(LSTM(150, return_sequences=False, kernel_regularizer=l2(0.001),recurrent_dropout=0.2,dropout=0.2))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(50, kernel_regularizer=l2(0.001)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(3, activation='softmax'))  # 3 classes: sell, neutral, buy
                
        # Compile the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['precision','recall'])
        self.model.summary()

        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_precision', mode='max',  patience=5, restore_best_weights=True)

        # Step 4: Train the Model
    
        lr_scheduler = ReduceLROnPlateau(monitor='val_precision', factor=0.5, patience=1)
        
        #Rolling Window Cross-Validation
        #hv-Blocked Cross-Validation
        def hv_block_split(X, y, n_splits, gap):
            tscv = TimeSeriesSplit(n_splits=n_splits)
            for train_index, test_index in tscv.split(X):
                train_index = train_index[:-gap]
                test_index = test_index[gap:]
                yield train_index, test_index

        n_splits = 5
        gap = 10
        
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(target_shift), y=target_shift)
        total_class_weight = class_weights.sum()
        class_weights_dict = {i : (class_weights[i]/total_class_weight) for i in range(len(class_weights))}
        print("class_weights_dict",class_weights_dict)

        for train_index, test_index in hv_block_split(X_scaled, y_encoded, n_splits, gap):

            X_train_inner, X_test_inner = X_scaled[train_index], X_scaled[test_index]
            y_train_inner, y_test_inner = y_encoded[train_index], y_encoded[test_index]
            
            # https://www.tensorflow.org/api_docs/python/tf/keras/Model
            history = self.model.fit(X_train_inner, y_train_inner, epochs=100, batch_size=64, validation_data=(X_test_inner, y_test_inner),  
                                 callbacks=[early_stopping,lr_scheduler],class_weight=class_weights_dict)
            
            y_pred = self.model.predict(X_test_inner)
            y_pred_classes = np.argmax(y_pred, axis=1)
            shifted_class_vector_y_pred = y_pred_classes - 1
            # Evaluate model
            # https://keras.io/api/models/model_training_apis/#evaluate-method
            val_accuracy = self.model.evaluate(X_test_inner, y_test_inner, verbose=0)
            print(val_accuracy)
            print(f"Validation Accuracy: {val_accuracy[1]}")
        
            ps = precision_score(np.argmax(y_test_inner, axis=1)- 1, shifted_class_vector_y_pred, average='weighted')
            print(f'Precission: {round(ps, 5)}')
            rs = recall_score(np.argmax(y_test_inner, axis=1)- 1, shifted_class_vector_y_pred,average='weighted')
            print(f'Recall: {round(rs, 5)}')
            f1s = f1_score(np.argmax(y_test_inner, axis=1)- 1, shifted_class_vector_y_pred, average='weighted')
            print(f'f1: {round(f1s, 5)}')

        self.model.save('lstm_stock_prediction_model.h5')
        #just split for overall test and check the scores
        _, X_test, _, y_test = train_test_split(X_scaled, y_encoded, test_size=0.4, shuffle=False)

        # Step 5: Evaluate and Predict
        y_pred = self.model.predict(X_test)
        
        #get class with heighest probability
        y_pred_classes = np.argmax(y_pred, axis=1)
        shifted_class_vector_y_pred = y_pred_classes - 1
        
        ps = precision_score(np.argmax(y_test, axis=1)- 1, shifted_class_vector_y_pred, average='weighted')
        print(f'Final Test Precission: {round(ps, 5)}')
        rs = recall_score(np.argmax(y_test, axis=1)- 1, shifted_class_vector_y_pred,average='weighted')
        print(f'Final Test Recall: {round(rs, 5)}')
        f1s = f1_score(np.argmax(y_test, axis=1)- 1, shifted_class_vector_y_pred, average='weighted')
        print(f'Final Test f1: {round(f1s, 5)}')
        return  self.model, scaler
       
    def predict_signals(self,model,scaler,df,features):

        df_features = df[features]
        X_test_scaled = scaler.transform(df_features)
        X_scaled = self.create_test_dataset( X_test_scaled , self.time_step)
      
       # X_test_scaled_new = self.create_test_dataset(X_test_scaled)
        y_pred = model.predict(X_scaled)
        #get class with heighest probability
        y_pred_classes = np.argmax(y_pred, axis=1)
         #back 0,1,2 to -1,0,1
        shifted_class_vector_y_pred = y_pred_classes - 1
        return pd.Series(shifted_class_vector_y_pred)
    
    def predict_signals_from_saved_model(self,df,features):
        with open('scaler_lstm.pkl','rb') as f:
            scaler = pickle.load(f)
        model = Sequential()
        model.load("lstm_stock_prediction_model_50_150_50_28_best_witout_DO.h5")
        df_features = df[features]
        X_test_scaled = scaler.transform(df_features)
        X_scaled = self.create_test_dataset( X_test_scaled , self.time_step)
      
        y_pred = model.predict(X_scaled)
        #get class with heighest probability
        y_pred_classes = np.argmax(y_pred, axis=1)
         #back 0,1,2 to -1,0,1
        shifted_class_vector_y_pred = y_pred_classes - 1
        return pd.Series(shifted_class_vector_y_pred)
