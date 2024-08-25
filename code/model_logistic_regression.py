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
        print(f'Precission: {round(precision_score(y_test, y_pred, average='macro'), 5)}')
        print(f'Recall: {round(recall_score(y_test, y_pred, average='macro'), 5)}')
        print(f'f1: {round(f1_score(y_test, y_pred, average='macro'), 5)}')

       
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
        print(f'Precission: {round(precision_score(y_test, y_pred, average='macro'), 5)}')
        print(f'Recall: {round(recall_score(y_test, y_pred, average='macro'), 5)}')
        print(f'f1: {round(f1_score(y_test, y_pred, average='macro'), 5)}')

       
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
           
    
    def build_model1(self, hp):
        model = Sequential()
        model.add(LSTM(
            units=hp.Int('units', min_value=10, max_value=100, step=10),
            input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
            return_sequences=False,
            kernel_regularizer=l2(hp.Float('l2', min_value=1e-5, max_value=1e-2, sampling='LOG'))))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.7, step=0.1)))
        model.add(Dense(3, activation='softmax', kernel_regularizer=l2(hp.Float('l2_dense', min_value=1e-5, max_value=1e-2, sampling='LOG'))))

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        
        return model
    def build_model2(self, hp):
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
    def train_lstm_model2(self, df_train,features, target, time_step=28):
        
        self.time_step = time_step
        df_features = df_train[features]
        target = df_train[target]  
        # Shift target values from [-1, 0, 1] to [0, 1, 2] and then encode
        target_shift = target + 1
        y_encoded = to_categorical(target_shift,num_classes = 3)  
        
        scaler = StandardScaler()
        X_scaled1 = scaler.fit_transform(df_features)

        X_scaled,y_encoded = self.create_train_dataset( X_scaled1, y_encoded, time_step)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.4, shuffle=False)
        self.input_shape = X_train.shape

        tuner = RandomSearch(
            self.build_model2,
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
        self.model.save('best_lstm_stock_prediction_model2.h5')

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

    def create_model(self, learning_rate=0.01, units=50, dropout_rate=0.2):
        model = Sequential()
        model.add(LSTM(units=units, return_sequences=True, input_shape=(1, 3)))
        model.add(LSTM(units=units))
        model.add(Dense(3, activation='softmax'))
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        return model
 
       
        df_features = df_train[features]
        target = df_train[target]  
        # Shift target values from [-1, 0, 1] to [0, 1, 2] and then encode
        target_shift = target + 1
        y_encoded = to_categorical(target_shift,num_classes = 3)  
        
        scaler = StandardScaler()
        X_scaled1 = scaler.fit_transform(df_features)


                # Define the parameter space
        param_distributions = {
            'units': [50],
            'learning_rate': uniform(0.001, 0.01),
            'dropout_rate': uniform(0.0, 0.5),
            'epochs': [10, 20, 30]
        }

        # Create an instance of your custom KerasClassifier
        model = KerasClassifier()

        # Set up RandomizedSearchCV
        random_search = RandomizedSearchCV(estimator=model, 
                                        param_distributions=param_distributions, 
                                        n_iter=10, 
                                        cv=TimeSeriesSplit(), 
                                        n_jobs=-1, 
                                        verbose=1,
                                        scoring='f1_macro')  # Using macro F1-score for imbalanced classes

        # Fit the model
        random_search.fit(X_scaled1, y_encoded)

    def train_lstm_model(self, df_train,features, target, time_step=28):
        self.time_step = time_step
        df_features = df_train[features]
        target = df_train[target]  
        # Shift target values from [-1, 0, 1] to [0, 1, 2] and then encode
        target_shift = target + 1
        #y_encoded = to_categorical(target_shift,num_classes = 3)  
        #print(to_categorical(target_shift,num_classes = 3).shape  )
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
      
        X_scaled,y_encoded = self.create_train_dataset( X_scaled1, y_encoded, time_step)
        #https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
      
        # Build the LSTM model (over fitting)
        self.model = Sequential()
        # self.model.add(LSTM(50, return_sequences=True, 
        #                     input_shape=(X_scaled.shape[1], X_scaled.shape[2]),
        #                      kernel_regularizer=l2(0.001),recurrent_dropout=0.2,dropout=0.2))#, kernel_regularizer=l2(0.001)

        # self.model.add(Dropout(0.5))
        # self.model.add(LSTM(150, return_sequences=False, kernel_regularizer=l2(0.001),recurrent_dropout=0.2,dropout=0.2))
        # self.model.add(Dropout(0.5))
        # self.model.add(Dense(50, kernel_regularizer=l2(0.001)))
        # self.model.add(Dropout(0.5))
        #self.add(BatchNormalization())
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
        #model.summary()

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

           # X_train_weight_inner, X_test_weight_inner = sample_weight[train_index], sample_weight[test_index]
            X_train_inner, X_test_inner = X_scaled[train_index], X_scaled[test_index]
            y_train_inner, y_test_inner = y_encoded[train_index], y_encoded[test_index]
            
            # https://www.tensorflow.org/api_docs/python/tf/keras/Model
            history = self.model.fit(X_train_inner, y_train_inner, epochs=100, batch_size=64, validation_data=(X_test_inner, y_test_inner),  
                                 callbacks=[early_stopping,lr_scheduler],class_weight=class_weights_dict)#class_weight=class_weights_dict ,sample_weight=X_train_weight_inner
            self.model.save('lstm_stock_prediction_model.h5')
            y_pred = self.model.predict(X_test_inner)
            y_pred_classes = np.argmax(y_pred, axis=1)
            shifted_class_vector_y_pred = y_pred_classes - 1
            # Evaluate model
            # https://keras.io/api/models/model_training_apis/#evaluate-method
            val_accuracy = self.model.evaluate(X_test_inner, y_test_inner, verbose=0)
            print(val_accuracy)
            print(f"Validation Accuracy: {val_accuracy[1]}")
            #print(f'Train Accuracy: {round(self.model.score(X_train_inner, y_train_inner), 5)}')
            #print(f'Test Accuracy: {round(self.model.score(X_test_inner, shifted_class_vector_y_pred), 5)}')

            print(f'Precission: {round(precision_score( np.argmax(y_test_inner, axis=1)- 1, shifted_class_vector_y_pred, average='macro'), 5)}')
            print(f'Recall: {round(recall_score(np.argmax(y_test_inner, axis=1)- 1, shifted_class_vector_y_pred, average='macro'), 5)}')
            print(f'f1: {round(f1_score(np.argmax(y_test_inner, axis=1)- 1, shifted_class_vector_y_pred, average='macro'), 5)}')

        #just split for overall test and check the scores
        _, X_test, _, y_test = train_test_split(X_scaled, y_encoded, test_size=0.4, shuffle=False)
        # Step 5: Evaluate and Predict
        y_pred = self.model.predict(X_test)
        #get class with heighest probability
        y_pred_classes = np.argmax(y_pred, axis=1)
        shifted_class_vector_y_pred = y_pred_classes - 1
        #y_test_classes = np.argmax(y_test, axis=1)
        
        #print(f'Final Test Accuracy: {round(accuracy_score(np.argmax(y_test, axis=1)- 1, shifted_class_vector_y_pred, average='macro'), 5)}')
        print(f'Final Test Precission: {round(precision_score( np.argmax(y_test, axis=1)- 1, shifted_class_vector_y_pred, average='macro'), 5)}')
        print(f'Final Test Recall: {round(recall_score(np.argmax(y_test, axis=1)- 1, shifted_class_vector_y_pred, average='macro'), 5)}')
        print(f'Final Test F1: {round(f1_score(np.argmax(y_test, axis=1)- 1, shifted_class_vector_y_pred, average='macro'), 5)}')
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
      
       # X_test_scaled_new = self.create_test_dataset(X_test_scaled)
        y_pred = model.predict(X_scaled)
        #get class with heighest probability
        y_pred_classes = np.argmax(y_pred, axis=1)
         #back 0,1,2 to -1,0,1
        shifted_class_vector_y_pred = y_pred_classes - 1
        return pd.Series(shifted_class_vector_y_pred)
    







class LongShortTermMemoryMLHelper_backup(): 
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
    def scheduler(self, epoch, lr):
        return 0.000000001
        if epoch < 50:
            return 0.0000001
        if(epoch % 10):
            return lr * 0.1
        
        if epoch < 200:
            return 0.0001 # Initial reduced learning rate
        else:
            #if lr>=0.0001:
                return lr * 0.1  # Further reduction after 10 epochs
            #else:
            #    return lr *0.1
            
    
    def build_model1(self, hp):
        model = Sequential()
        model.add(LSTM(
            units=hp.Int('units', min_value=10, max_value=100, step=10),
            input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
            return_sequences=False,
            kernel_regularizer=l2(hp.Float('l2', min_value=1e-5, max_value=1e-2, sampling='LOG'))))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.7, step=0.1)))
        model.add(Dense(3, activation='softmax', kernel_regularizer=l2(hp.Float('l2_dense', min_value=1e-5, max_value=1e-2, sampling='LOG'))))

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        
        return model
    def build_model2(self, hp):
        model = Sequential()
    
        # Tune the number of LSTM layers
        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(LSTM(
                units=hp.Int(f'units_{i}', min_value=10, max_value=100, step=10),
                return_sequences=True if i < hp.get('num_layers') - 1 else False,
                input_shape=(self.X_train.shape[1], self.X_train.shape[2]) if i == 0 else None,
                kernel_regularizer=l2(hp.Float(f'l2_{i}', min_value=1e-5, max_value=1e-2, sampling='LOG'))))
            model.add(BatchNormalization())
            model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.7, step=0.1)))
        
        model.add(Dense(3, activation='softmax', kernel_regularizer=l2(hp.Float('l2_dense', min_value=1e-5, max_value=1e-2, sampling='LOG'))))

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        
        return model
    def train_lstm_model2(self, df_train,features, target, time_step=28):
        

        self.X_train, self.X_test, self.y_train, self.y_test, self.class_weights_dict = self.train_lstm_model(df_train,features, target, time_step)

        tuner = RandomSearch(
            self.build_model2,
            objective='accuracy',
            max_trials=20,
            executions_per_trial=2,
            directory='hyperparameter_tuning_model2',
            project_name='lstm_stock_prediction')
        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Learning rate scheduler
        lr_scheduler = LearningRateScheduler(self.scheduler)

        # Perform the hyperparameter search
        tuner.search(self.X_train, self.y_train, epochs=50, validation_data=(self.X_test, self.y_test), 
                    callbacks=[early_stopping, lr_scheduler], batch_size=1024,class_weight={0:.55,1:0.05,2:0.55})

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
        history =  self.model.fit(self.X_train, self.y_train, epochs=100, batch_size=1024, validation_data=(self.X_test, self.y_test), 
                            callbacks=[early_stopping, lr_scheduler])

        # Evaluate the model
        loss, accuracy =  self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Test Accuracy: {accuracy:.2f}")

        # Save the model
        self.model.save('best_lstm_stock_prediction_model2.h5')

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

    def train_lstm_model(self, df_train,features, target, time_step=28):


        X_train, X_test, y_train, y_test,X_undersampled_scaled, y_undersampled_encoded,X_undersampled, y_undersampled = self.combine_dataset(df_train,time_step)#self.under_sample(df_train,features, target,self.scaler) 
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=False)

        y_train_integer = np.argmax(y_train, axis=1)
        #print(len(X_train_scaled))
        #print(pd.Series(y_train_integer).value_counts())
        # Calculate class weights
        # Calculate class weights
        #class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train_integer), y_train_integer)
        class_weights = class_weight.compute_class_weight(class_weight=None, classes=np.unique(y_train_integer), y=y_train_integer)

        # Convert class weights to dictionary for use in training
        class_weights_dict = {i : class_weights[i] for i in range(len(class_weights))}
        #class_weights_dict = {0:0.45,1:0.1,2:0.45}
        #print()
        #print(class_weights_dict)
        #return X_train, X_test, y_train, y_test, class_weights_dict
        #https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
        # self.model = Sequential()
        # self.model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X.shape[2]), activation = 'relu'))
        # self.model.add(Dropout(0.2))
        # self.model.add(LSTM(25, return_sequences=False, activation = 'relu'))
        # self.model.add(Dropout(0.3))
        # self.model.add(Dense(15, activation = 'relu'))
        # self.model.add(Dropout(0.2))
        # self.model.add(Dense(3, activation='softmax'))

        # Build the LSTM model (over fitting)
        self.model = Sequential()
        #print("X_train.shape",X_train.shape)
        #print("y_train.shape",y_train.shape)
        #print(X_train[0])
        #self.model.add(BatchNormalization(input_shape=(self.dataset1[0]["X_train"].shape[1], self.dataset1[0]["X_train"].shape[2])))
    
        self.model.add(LSTM(10, return_sequences=True, 
                            input_shape=(X_train.shape[1], X_train.shape[2]),
                             kernel_regularizer=l2(0.001),recurrent_dropout=0.7,dropout=0.5))#, kernel_regularizer=l2(0.001)

        self.model.add(Dropout(0.5))
        self.model.add(LSTM(10, return_sequences=False, kernel_regularizer=l2(0.001),recurrent_dropout=0.7,dropout=0.5))
        self.model.add(Dropout(0.5))
        #self.model.add(Dense(250, kernel_regularizer=l2(0.001)))
       # self.model.add(Dropout(0.5))
        #self.model.add(Dense(3, activation='softmax'))
        #self.model.add(Dropout(0.5))
        # self.model.add(GRU(64, return_sequences=False))
        # self.model.add(BatchNormalization())
        # self.model.add(Dropout(0.5))
        #self.model.add(GRU(64, return_sequences=False, kernel_regularizer=l2(0.001)))# kernel_regularizer=l2(0.001)
        #self.model.add(Dropout(0.5))
        #self.model.add(Dense(10, kernel_regularizer=l2(0.001), activation='relu'))
        #self.model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        #self.model.add(Dropout(0.5))


        self.model.add(Dense(3, activation='softmax'))  # 3 classes: sell, neutral, buy


        # self.model = Sequential() #(littel better but still over fit)
        # self.model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
        # self.model.add(Dropout(0.3))  # Increased dropout rate
        # self.model.add(LSTM(50))
        # self.model.add(Dropout(0.3))  # Increased dropout rate
        # self.model.add(Dense(3, activation='softmax'))  # 3 classes: sell, neutral, buy

        # # Build the LSTM model with increased regularization
        # self.model = Sequential()
        # self.model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, activation = 'relu'))
        # self.model.add(Dropout(0.5))  # Slightly increased dropout rate
        # self.model.add(LSTM(50,activation = 'relu') )
        # self.model.add(Dropout(0.5))  # Slightly increased dropout rate
        # self.model.add(Dense(25, activation='relu'))  # 3 classes: sell, neutral, buy
        # self.model.add(Dropout(0.5))  # Slightly increased dropout rate
        # self.model.add(Dense(3, activation='softmax'))  # 3 classes: sell, neutral, buy
        # self.model = Sequential()
        # self.model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X.shape[2]), activation = 'relu'))
        # self.model.add(Dropout(0.5))
        # self.model.add(LSTM(50, return_sequences=False, activation = 'relu'))
        # self.model.add(Dropout(0.3))
        # self.model.add(Dense(25, activation = 'relu'))
        # self.model.add(Dropout(0.2))
        # self.model.add(Dense(3, activation='softmax'))

        # #Build the LSTM model with simplified architecture and increased dropout
        # self.model = Sequential()
        # self.model.add(LSTM(5 , input_shape=(X_train.shape[1], X_train.shape[2])))  # Reduced number of units
        # self.model.add(Dropout(0.6))  # Increased dropout rate
        # self.model.add(Dense(3, activation='softmax'))  # 3 classes: sell, neutral, buy

        # # Step 3: Build the LSTM Model
        # self.model = Sequential()
        # self.model.add(LSTM(200,return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=l2(0.02)))
        # self.model.add(BatchNormalization())
        # self.model.add(Dropout(0.6)) 
        # self.model.add(LSTM(100, return_sequences=False, kernel_regularizer=l2(0.02)))
        # self.model.add(Dropout(0.6)) 
        # self.model.add(Dense(50, kernel_regularizer=l2(0.01)))
        # self.model.add(Dropout(0.5)) 
        # self.model.add(Dense(25, kernel_regularizer=l2(0.01)))
        # self.model.add(Dropout(0.6))
        # self.model.add(Dense(3, activation='softmax'))

        # self.model.add(tf.keras.layers.LSTM(units=hp.Int('units',min_value=40, max_value=800, step=20),
        #                       dropout=hp.Float('droput',min_value=0.15, max_value=0.99, step=0.05),
        #                       recurrent_dropout=hp.Float('redroput',min_value=0.05, max_value=0.99, step=0.05),
        #                       activation='relu',
        #                       return_sequences=True,input_shape=(X_train.shape[1], X_train.shape[2])))
                
        # Compile the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['precision','recall'])
        #model.summary()

        # Use SMOTE to oversample the minority classes
        
        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_precision', mode='max',  patience=5, restore_best_weights=True)

        # Step 4: Train the Model
    
        lr_callback = LearningRateScheduler(self.scheduler)
        lr_scheduler = ReduceLROnPlateau(monitor='val_precision', factor=0.5, patience=1)
        
        #Rolling Window Cross-Validation
        n_splits = 3
        tscv = TimeSeriesSplit(n_splits=n_splits)
        #X_undersampled, y_undersampled
        _X_undersampled_scaled,_y_undersampled_encoded = self.create_train_dataset(X_undersampled_scaled, y_undersampled_encoded, time_step)
        #for train_index, test_index in tscv.split(_X_undersampled_scaled):
        initial_window = 365*2  # Initial training set size
        step_size = 365  # Step size for expanding the window
        #hv-Blocked Cross-Validation
        def hv_block_split(X, y, n_splits, gap):
            tscv = TimeSeriesSplit(n_splits=n_splits)
            for train_index, test_index in tscv.split(X):
                train_index = train_index[:-gap]
                test_index = test_index[gap:]
                yield train_index, test_index
        n_splits = 4
        gap = 10
       
        #for start in range(initial_window, len(_X_undersampled_scaled) - step_size, step_size):
        #    end = start + step_size
        #class_weightss = pd.Series(np.argmax(_y_undersampled_encoded, axis=1)).value_counts().to_dict()
        #sample_weight =  pd.Series(np.argmax(_y_undersampled_encoded, axis=1)).map(lambda x: 1/class_weightss[x])
        #print("class_weightss", pd.Series(np.argmax(_y_undersampled_encoded, axis=1)).value_counts().to_dict())
        tmp_y_undersampled = pd.Series(y_undersampled+1)
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(tmp_y_undersampled), y=tmp_y_undersampled)
        total_class_weight = class_weights.sum()
        class_weights_dict = {i : (class_weights[i]/total_class_weight) for i in range(len(class_weights))}
        print("class_weights_dict",class_weights_dict)
        scaler = StandardScaler()
        X_undersampled_scaled1 = scaler.fit_transform(X_undersampled)
        y_undersampled_encoded = to_categorical(y_undersampled + 1,num_classes = 3)  # Shift target values from [-1, 0, 1] to [0, 1, 2]
        X_undersampled_scaled1,y_undersampled_encoded = self.create_train_dataset(X_undersampled_scaled1, y_undersampled_encoded, time_step)
        #for train_index, test_index in tscv.split(_X_undersampled_scaled):
        #for start in range(initial_window, len(_X_undersampled_scaled) - step_size, step_size):
        #    end = start + step_size
        #https://python.plainenglish.io/cross-validation-techniques-for-time-series-data-d1ad7a3a680b
        for train_index, test_index in hv_block_split(_X_undersampled_scaled, _y_undersampled_encoded, n_splits, gap):
            
            
           # print("X_undersampled_scaled1",len(X_undersampled_scaled1))
            #print("y_undersampled_encoded",len(y_undersampled_encoded))
           


            X_train_inner, X_test_inner = X_undersampled_scaled1[train_index], X_undersampled_scaled1[test_index]
            y_train_inner, y_test_inner = y_undersampled_encoded[train_index], y_undersampled_encoded[test_index]
            
            #X_train_inner, X_test_inner = X_undersampled_scaled1[:start,:], X_undersampled_scaled1[start:end,:]
            #y_train_inner, y_test_inner = y_undersampled_encoded[:start,:], y_undersampled_encoded[start:end,:]

            #X_train_inner, X_test_inner = X_undersampled_scaled1[train_index], X_undersampled_scaled1[test_index]
            #y_train_inner, y_test_inner = y_undersampled_encoded[train_index], y_undersampled_encoded[test_index]

            #class_weights = class_weight.compute_class_weight(class_weight=None, classes=np.unique((y_undersampled + 1)), y=(y_undersampled + 1))

            # Convert class weights to dictionary for use in training
            #class_weights_dict = {i : class_weights[i] for i in range(len(class_weights))}
           
            #print(sample_weight)
            # https://www.tensorflow.org/api_docs/python/tf/keras/Model
            history = self.model.fit(X_train_inner, y_train_inner, epochs=100, batch_size=64, validation_data=(X_test_inner, y_test_inner),  
                                 callbacks=[early_stopping,lr_scheduler],class_weight=class_weights_dict)#class_weight=class_weights_dict ,
            self.model.save('lstm_stock_prediction_model.h5')
            y_pred = self.model.predict(X_test_inner)
            y_pred_classes = np.argmax(y_pred, axis=1)
            shifted_class_vector_y_pred = y_pred_classes - 1
            # Evaluate model
            # https://keras.io/api/models/model_training_apis/#evaluate-method
            val_accuracy = self.model.evaluate(X_test_inner, y_test_inner, verbose=0)
            print(val_accuracy)
            print(f"Validation Accuracy: {val_accuracy[1]}")
            #print(f'Train Accuracy: {round(self.model.score(X_train_inner, y_train_inner), 5)}')
            #print(f'Test Accuracy: {round(self.model.score(X_test_inner, shifted_class_vector_y_pred), 5)}')

            print(f'Precission: {round(precision_score( np.argmax(y_test_inner, axis=1)- 1, shifted_class_vector_y_pred, average='macro'), 5)}')
            print(f'Recall: {round(recall_score(np.argmax(y_test_inner, axis=1)- 1, shifted_class_vector_y_pred, average='macro'), 5)}')
            print(f'f1: {round(f1_score(np.argmax(y_test_inner, axis=1)- 1, shifted_class_vector_y_pred, average='macro'), 5)}')
            # print("Precission: " + round(precision_score(y_test, y_pred, average='macro')))
            # print("Recall: " + round(recall_score(y_test, y_pred, average='macro')))
            # print("f1: " + round(f1_score(y_test, y_pred, average='macro')))


        # Step 5: Evaluate and Predict
        y_pred = self.model.predict(X_test)
        #get class with heighest probability
        y_pred_classes = np.argmax(y_pred, axis=1)
        shifted_class_vector_y_pred = y_pred_classes - 1
        #y_test_classes = np.argmax(y_test, axis=1)
        print(f'Final Precission: {round(precision_score( np.argmax(y_test, axis=1)- 1, shifted_class_vector_y_pred, average='macro'), 5)}')
        print(f'Final Recall: {round(recall_score(np.argmax(y_test, axis=1)- 1, shifted_class_vector_y_pred, average='macro'), 5)}')
        print(f'Final f1: {round(f1_score(np.argmax(y_test, axis=1)- 1, shifted_class_vector_y_pred, average='macro'), 5)}')
        return  self.model, scaler
       

        #back 0,1,2 to -1,0,1
        #shifted_class_vector_y_pred = y_pred_classes - 1
       # shifted_class_vector_y_test = y_test_classes - 1
        #print("-----------predicated----")
        #print(pd.Series(shifted_class_vector_y_pred).value_counts())
        #print("----------test actual-----")
        #print(pd.Series(shifted_class_vector_y_test).value_counts())
        #print("---------------")
        #SSSSprint(y_test_classes)
        # Plot actual vs predicted signals

        # # Plot training and validation loss
        # plt.plot(history.history['loss'], label='Training Loss')
        # plt.plot(history.history['val_loss'], label='Validation Loss')
        # plt.title('Training and Validation Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # #plt.xticks(np.arange(0, len(history.history['loss'])+1, 1.0))
        # plt.show()

        
        # plt.plot(history.history['precision'], label='precision')
        # plt.plot(history.history['recall'], label='recall')
        # plt.title('Training and Validation Accuracy')
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.legend()
        # plt.show()


        # # return self.model, self.scaler
        #return 1, 2
    def predict_signals(self,model,scaler,df,features):

        df_features = df[features]

        X_test_scaled = scaler.transform(df_features)
        X_test_scaled_new = self.create_test_dataset(X_test_scaled)
        y_pred = model.predict(X_test_scaled_new)
        #get class with heighest probability
        y_pred_classes = np.argmax(y_pred, axis=1)
         #back 0,1,2 to -1,0,1
        shifted_class_vector_y_pred = y_pred_classes - 1
        return pd.Series(shifted_class_vector_y_pred)
    
    def predict_signals2(self,ticker,df,features):
        df_features = df[features]

        for i in range(len(self.dataset1)):
            if self.dataset1[i]["ticker"]==ticker:
                scaler=self.dataset1[i]["scaler"]
        X_test_scaled = scaler.transform(df_features)
        X_test_scaled_new = self.create_test_dataset(X_test_scaled)
        y_pred = self.model.predict(X_test_scaled_new)
        #get class with heighest probability
        y_pred_classes = np.argmax(y_pred, axis=1)
         #back 0,1,2 to -1,0,1
        shifted_class_vector_y_pred = y_pred_classes - 1
        return pd.Series(shifted_class_vector_y_pred)
    def add_dataset_under_sample(self, ticker, data_combined,features,target,time_step):
         # Separate majority and minority classes
        majority_class = data_combined[data_combined[target] == 0]
        minority_class_buy = data_combined[data_combined[target] == 1]
        minority_class_sell = data_combined[data_combined[target] == -1]
        minority_len = max(len(minority_class_buy), len(minority_class_sell))

        # Undersample the majority class
        
       # majority_class_undersampled = majority_class[len(majority_class)-minority_len-1:] #resample(majority_class, replace=False, n_samples=len(minority_class_buy), shuffle=False)
        majority_class_undersampled = majority_class#[:minority_len*5] 
        #print("before",data_combined[target].value_counts())
        # Combine the undersampled majority class with the minorit,scalery classes
        undersampled_data = pd.concat([majority_class_undersampled, minority_class_buy, minority_class_sell],ignore_index=False)
      
        #undersampled_data = data_combined.drop(majority_class[:math.ceil((len(majority_class)-minority_len)/3)].index)
        #print("after",undersampled_data[target].value_counts())
        scaler = StandardScaler()
        X_train, X_test, y_train, y_test,X_undersampled_scaled, y_undersampled_encoded, X_undersampled, y_undersampled = self.under_sample(undersampled_data,features,target,scaler,time_step)
        self.dataset1.append({"ticker":ticker,"scaler":scaler,"undersampled_data":undersampled_data
                                ,"X_train":X_train, "X_test":X_test, "y_train":y_train, "y_test":y_test,
                                "X_undersampled_scaled":X_undersampled_scaled,"y_undersampled_encoded":y_undersampled_encoded,
                                "X_undersampled":X_undersampled, "y_undersampled":y_undersampled
                                })

        # Shuffle the data
        #undersampled_data = undersampled_data.sample(frac=1)#.reset_index(drop=True)
        #return undersampled_data

    def scale_by_window(self, data, window_size, feature_range=(0, 1)):
        scaler = MinMaxScaler(feature_range=feature_range)
        scaled_data = np.zeros_like(data)
        
        for start in range(0, len(data) - window_size + 1):
            end = start + window_size
            window = data[start:end]
            scaled_window = scaler.fit_transform(window)
            scaled_data[start:end] = scaled_window
            
        return scaled_data
    def scale_feature(self,data,window_size):
        # Scale each feature by window
        scaled_features = []
        for i in range(data.shape[1]):
            feature_data = data[:, i].reshape(-1, 1)  # Select each feature column
            scaled_feature = self.scale_by_window(feature_data, window_size)
            scaled_features.append(scaled_feature)


    def under_sample(self, undersampled_data,features,target,scaler,time_step):
        feature_length = len(features)

       
        #X_test, y_test = self.create_train_dataset(X_test, y_test, time_step)

        # # Separate majority and minority classes
        # majority_class = data_combined[data_combined[target] == 0]
        # minority_class_buy = data_combined[data_combined[target] == 1]
        # minority_class_sell = data_combined[data_combined[target] == -1]

        # # Undersample the majority class
        # majority_class_undersampled = majority_class[:len(minority_class_buy)] #resample(majority_class, replace=False, n_samples=len(minority_class_buy), shuffle=False)

        # # Combine the undersampled majority class with the minorit,scalery classes
        # undersampled_data = pd.concat([majority_class_undersampled, minority_class_buy, minority_class_sell],ignore_index=False)
       
        # Shuffle the data
        #undersampled_data = undersampled_data.sample(frac=1)#.reset_index(drop=True)

        # Separate features and target
        X_undersampled = undersampled_data[features].values
        y_undersampled = undersampled_data[target].values
        
        #X_train_log = np.log1p(X_undersampled)
        # Normalize the features using StandardScaler
        X_undersampled_scaled = scaler.fit_transform(X_undersampled)

        # One-hot encode the target
        y_undersampled_encoded = to_categorical(y_undersampled + 1,num_classes = 3)  # Shift target values from [-1, 0, 1] to [0, 1, 2]
        X_undersampled_scaled,y_undersampled_encoded = self.create_train_dataset(X_undersampled_scaled, y_undersampled_encoded, time_step)
        ## Split the data into training and testing sets
        split_index = int(len(undersampled_data) * 0.70)

        ##Split the data into training and testing sets
        X_train = X_undersampled_scaled[:split_index,:]
        X_test = X_undersampled_scaled[split_index:,:]
        y_train = y_undersampled_encoded[:split_index,:]
        y_test = y_undersampled_encoded[split_index:,:]

        #X_train, X_test, y_train, y_test = train_test_split(X_undersampled_scaled, y_undersampled_encoded,
        #                                                     test_size=0.3,shuffle=False )#shuffle=False ,stratify=y_undersampled_encoded

        # #print("y_train")
        # #print(pd.Series(np.argmax(y_train, axis=1)-1 ).value_counts())
        
        # #print("y_test")
        # #print(pd.Series(np.argmax(y_test, axis=1)-1 ).value_counts())
        # # # Reshape input to be 3D [samples, timesteps, features]
     
        # # Calculate how many rows need to be trimmed from the beginning
        # total_elements_x_train = X_train.shape[0]
        # rows_to_trim_x_train = total_elements_x_train % time_step
        # # Trim the starting rows to make the array shape compatible for reshaping
        # trimmed_array_x_train = X_train[rows_to_trim_x_train:]

        # # Calculate the new value of n
        # n_train = trimmed_array_x_train.shape[0] // time_step
        # # Reshape the trimmed array to (n, i, 4)
        # X_train = trimmed_array_x_train.reshape(-1, time_step, feature_length)
        #X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        #X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
       
        # # Calculate how many rows need to be trimmed from the beginning
        # total_elements_y_test = X_test.shape[0]
        # rows_to_trim_x_test = total_elements_y_test % time_step
 
        # #rows_to_trim = total_elements % (4*time_step)
        # # Trim the starting rows to make the array shape compatible for reshaping
        # trimmed_array_x_test = X_test[rows_to_trim_x_test:]

        # # Calculate the new value of n
        # n_test = trimmed_array_x_test.shape[0] // time_step
        # # Reshape the trimmed array to (n, i, 4)
     
        # X_test = trimmed_array_x_test.reshape(-1, time_step, feature_length)

     
        # #X_test = X_test.reshape((-1, time_step, X_test.shape[1]))
    

        # trimmed_array_y_train = y_train[rows_to_trim_x_train:]
        # y_train = trimmed_array_y_train[::time_step]

        # trimmed_array_y_test = y_test[rows_to_trim_x_test:]
        # y_test = trimmed_array_y_test[::time_step]
  


        #print("before", X_train.shape)
        #X_train,y_train = self.create_train_dataset(X_train, y_train, time_step)
        #X_test, y_test = self.create_train_dataset(X_test, y_test, time_step)
       # print("after", X_train.shape)

        return  X_train, X_test, y_train, y_test, X_undersampled_scaled, y_undersampled_encoded, X_undersampled, y_undersampled
    def combine_dataset(self,tickerlist,time_step):
        X_train=np.array([])
        X_test=np.array([])
        y_train=np.array([])
        y_test=np.array([])
        cnt = 0
        for i in range(len(self.dataset1)):
            if(len(tickerlist) == 0 or any(self.dataset1[i]["ticker"] in x for x in tickerlist)):
                if cnt==0:
                    X_train=self.dataset1[i]["X_train"]
                    X_test=self.dataset1[i]["X_test"]
                    y_train=self.dataset1[i]["y_train"]
                    y_test=self.dataset1[i]["y_test"]
                    X_undersampled_scaled=self.dataset1[i]["X_undersampled_scaled"]
                    y_undersampled_encoded=self.dataset1[i]["y_undersampled_encoded"]
                    X_undersampled=self.dataset1[i]["X_undersampled"]
                    y_undersampled=self.dataset1[i]["y_undersampled"]
                else:
                
                    y_train=np.concatenate((y_train,self.dataset1[i]["y_train"]), axis=0)
                    X_train=np.concatenate((X_train,self.dataset1[i]["X_train"]), axis=0)
                    X_test=np.concatenate((X_test,self.dataset1[i]["X_test"]), axis=0)

                    y_test=np.concatenate((y_test,self.dataset1[i]["y_test"]), axis=0)
                    X_undersampled_scaled=np.concatenate((y_test,self.dataset1[i]["X_undersampled_scaled"]), axis=0)
                    y_undersampled_encoded=np.concatenate((y_test,self.dataset1[i]["y_undersampled_encoded"]), axis=0)
                    X_undersampled=np.concatenate((y_test,self.dataset1[i]["X_undersampled"]), axis=0)
                    y_undersampled=np.concatenate((y_test,self.dataset1[i]["y_undersampled"]), axis=0)
                cnt=cnt+1
        #X_train,y_train = self.create_train_dataset(X_train, y_train, time_step)
        #X_test, y_test = self.create_train_dataset(X_test, y_test, time_step)

        return  X_train, X_test, y_train, y_test,X_undersampled_scaled, y_undersampled_encoded, X_undersampled, y_undersampled

