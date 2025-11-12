
from components.preprocessing import Preproccessor
from components.training import Regression_Training,Classification_Training


dataset_path = "datasets/regression/aqi.csv"
preprocessor = Preproccessor(dataset_path, "aqi_value")
X_train, y_train, X_test, y_test, X_val, y_val, task_type = (
    preprocessor.run_preprocessing()
)
trainer = Regression_Training(X_train, y_train, X_test, y_test, X_val, y_val,dataset_path,"aqi_value")
trainer.train_model()


# dataset_path = "datasets/regression/aqi.csv"
# preprocessor = Preproccessor(dataset_path, "aqi_value")
# X_train, y_train, X_test, y_test, X_val, y_val, task_type = (
#     preprocessor.run_preprocessing()
# )
# trainer = Classification_Training(X_train, y_train, X_test, y_test, X_val, y_val,dataset_path=dataset_path,target_col='target')
# trainer.train_models()
  
# trainer.tune_models()
  