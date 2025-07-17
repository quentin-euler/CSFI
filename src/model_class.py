import geopandas as gpd
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef

class Model:
    def __init__(self, dataset_path = None, train = False, predict = True, config  = None, **kwargs):
        self.model_config = self._init_config()
        self.change_config(config, **kwargs)   
        self.dataset = None 
        if dataset_path :
            self.dataset = gpd.read_file(dataset_path)
        self.dataset_path = dataset_path
        self.predictor = self._init_predictor()
        self.predictions = None
        if train:
            self.train()
            if predict:
                self.predictions = self.predictor(self.dataset)


    def __call__(self, input_data):
        predictions = self.predictor(input_data)
        self.predictions = input_data.copy()
        self.predictions["predicted"] = predictions["predicted"].astype(bool)
        if "proba" in predictions:
            self.predictions["proba"] = predictions["proba"]
        return predictions

    def train(self, test_size = 0.2):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def _init_predictor(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def _init_config(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def change_config(self, config =None, **kwargs):
        """
        Change the model configuration.
        """
        if config:
            self.model_config = config
        else:
            for key, value in kwargs.items():
                if key in self.model_config:
                    self.model_config[key]["value"] = value
                else:
                    print(f"Invalid configuration key: {key}")
    
    def get(self, param):
        """
        Get the value of a specific parameter from the model configuration.
        """
        if param in self.model_config:
            return self.model_config[param]["value"]
        else:
            raise KeyError(f"Parameter '{param}' not found in model configuration.")
    
    def explore_predictions(self):
        return self.predictions.explore(
            column="predicted",
            cmap="RdYlGn",
            legend=True,
            name="Predicted Flooded Areas",
            tooltip=["flooded", "predicted"],
        )
    
    def analyse_predictions(self):
        """
        Analyse the predictions made by the model.
        """
        gt = self.dataset["flooded"]
        pred = self.predictions["predicted"]
        
        f1 = f1_score(gt, pred)
        acc = accuracy_score(gt, pred)
        corr = matthews_corrcoef(gt, pred)
        
        return {
            "f1_score": f1,
            "accuracy": acc,
            "correlation": corr
        }
    
    def train_test(self, train_data, test_data, from_path = False):
        """
        Train the model on the training data and evaluate it on the test data.
        """
        if from_path:
            train_data = gpd.read_file(train_data)
            test_data = gpd.read_file(test_data)
  
        self.dataset = train_data
        
        self.train(test_size = 0)
        predictions = self.predictor(test_data)
        
        self.predictions = test_data.copy()
        self.predictions["predicted"] = predictions["predicted"]
        self.predictions["flooded"] = test_data["flooded"]
        
        gt = test_data["flooded"]
        pred = predictions["predicted"]
        
        f1 = f1_score(gt, pred)
        acc = accuracy_score(gt, pred)
        corr = matthews_corrcoef(gt, pred)
        
        return {
            "f1_score": f1,
            "accuracy": acc,
            "correlation": corr
        }
