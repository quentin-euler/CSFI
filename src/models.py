import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef

import xgboost as xgb
from tqdm import tqdm
import torch
import torch.nn as nn
from model_class import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import json
import os



class NDWIMeanModel(Model):
    def __init__(self, **kwargs):
        """
        Naive predictor that uses the mean NDWI to predict floods.
        """
        super().__init__(**kwargs)
    
    def train(self, test_size = 0.2):
        self.__call__(self.dataset)
        thresholds = np.arange(-1, 1.05, 0.05)
        best_score = 0
        best_threshold = -1
        for threshold in tqdm(thresholds, desc="Finding optimal threshold"):
            if self.predictions is not None:
                self.predictions["predicted"] = self.predictions["NDWI"] > threshold
                self.predictions["predicted"] = self.predictions["predicted"].astype(bool)
            results = self.analyse_predictions()
            score = results["accuracy"]
            if score > best_score:
                best_score = score
                best_threshold = threshold
        self.change_config(threshold=best_threshold)
        if self.predictions is not None:
            self.predictions["predicted"] = self.predictions["NDWI"] > best_threshold
            self.predictions["predicted"] = self.predictions["predicted"].astype(bool)

        
    def _init_predictor(self):
        """
        Initialize the predictor. For the naive predictor, this is just a placeholder.
        """
        def f(dataset):
            threshold = self.get("threshold")
            output = dataset.copy()
            output["predicted"] = output["NDWI"] > threshold
            output["predicted"] = output["predicted"].astype(bool)
            return output
        return f
        
    def _init_config(self):
        """
        Initialize the model configuration.
        """
        return {
            "name": "NDWIMeanModel",
            "description": "Predicts floods based on the mean NDWI value.",
            "threshold": param_config(0, True, "float", -1.0, 1.0, log_scale=False)
        }


        
        
class XGBoostModel(Model):
    def __init__(self, **kwargs):
        self.model = xgb.XGBClassifier()
        super().__init__(**kwargs)
    
    def train(self, test_size = 0.2):
        """
        Train the model using the dataset.
        """
        max_depth = self.get("max_depth")
        n_estimators = self.get("n_estimators")
        subsample = self.get("subsample")
        learning_rate = self.get("learning_rate")
        
        self.model = xgb.XGBClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            subsample=subsample,
            learning_rate=learning_rate,
            eval_metric='logloss'
        )
        
        if self.dataset is None:
            raise ValueError("No dataset to train on. Please add data first.")
        
        # Drop rows with NaN values in the selected parameters or target
        self.dataset = self.dataset.dropna()
        features_col = [col for col in self.dataset.columns if col.startswith("mean_band_") or col in ["mean", "std", "min", "max"]]
        X = self.dataset[features_col].values
        y = self.dataset["flooded"].astype(int).values


        if test_size > 0:
            X, X_test, y, y_test = train_test_split(X, y, test_size=test_size)
        
        self.model.fit(X, y)

        if test_size > 0:
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            corr = matthews_corrcoef(y_test, y_pred)
            
            self.y_pred = y_pred
            self.y_true = y_test
            
            self.training_results = {
                "accuracy": acc,
                "f1_score": f1,
                "correlation": corr
            }
        
    def _init_predictor(self):
        def f(dataset):
            """
            Predicts floods using the trained model.
            """
            if self.model is None:
                raise ValueError("Model has not been trained yet.")
            output = dataset.copy()
            features_col = [col for col in dataset.columns if col.startswith("mean_band_")]
            X = output[features_col].values
            output["predicted"] = self.model.predict(X)
            return output
        return f
    
    def _init_config(self):
        """
        Initialize the model configuration.
        """
        return {
            "name": "XGBoostModel",
            "description": "Predicts floods using an XGBoost classifier.",
            "max_depth": param_config(3, True, "int", 1, 10),
            "n_estimators": param_config(300, True, "int", 100, 1_000),
            "subsample": param_config(0.8, True, "float", 0.5, 1.0),
            "learning_rate": param_config(0.1, True, "float", 0.001, 0.5, log_scale=True),
        }
    
class RandomForestModel(Model):
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier()
        super().__init__(**kwargs)

    def train(self, test_size=0.2):
        n_estimators = self.get("n_estimators")
        max_depth = self.get("max_depth")
        min_samples_split = self.get("min_samples_split")
        min_samples_leaf = self.get("min_samples_leaf")

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )

        if self.dataset is None:
            raise ValueError("No dataset to train on. Please add data first.")

        self.dataset = self.dataset.dropna()
        features_col = [col for col in self.dataset.columns if col.startswith("mean_band_") or col in ["mean", "std", "min", "max"]]
        X = self.dataset[features_col].values
        y = self.dataset["flooded"].astype(int).values

        if test_size > 0:
            X, X_test, y, y_test = train_test_split(X, y, test_size=test_size)

        self.model.fit(X, y)

        if test_size > 0:
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            corr = matthews_corrcoef(y_test, y_pred)
            self.y_pred = y_pred
            self.y_true = y_test
            self.training_results = {
                "accuracy": acc,
                "f1_score": f1,
                "correlation": corr
            }

    def _init_predictor(self):
        def f(dataset):
            if self.model is None:
                raise ValueError("Model has not been trained yet.")
            output = dataset.copy()
            features_col = [col for col in dataset.columns if col.startswith("mean_band_")]
            X = output[features_col].values
            output["predicted"] = self.model.predict(X)
            return output
        return f

    def _init_config(self):
        return {
            "name": "RandomForestModel",
            "description": "Predicts floods using a Random Forest classifier.",
            "n_estimators": param_config(100, True, "int", 10, 500),
            "max_depth": param_config(None, True, "int", 1, 20),
            "min_samples_split": param_config(2, True, "int", 2, 10),
            "min_samples_leaf": param_config(1, True, "int", 1, 10),
        }


class SVMModel(Model):
    def __init__(self, **kwargs):
        self.model = SVC(probability=False)
        super().__init__(**kwargs)

    def train(self, test_size=0.2):
        C = self.get("C")
        kernel = self.get("kernel")

        self.model = SVC(kernel=kernel, C=C)

        if self.dataset is None:
            raise ValueError("No dataset to train on. Please add data first.")

        self.dataset = self.dataset.dropna()
        features_col = [col for col in self.dataset.columns if col.startswith("mean_band_") or col in ["mean", "std", "min", "max"]]
        X = self.dataset[features_col].values
        y = self.dataset["flooded"].astype(int).values

        if test_size > 0:
            X, X_test, y, y_test = train_test_split(X, y, test_size=test_size)

        self.model.fit(X, y)

        if test_size > 0:
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            corr = matthews_corrcoef(y_test, y_pred)
            self.y_pred = y_pred
            self.y_true = y_test
            self.training_results = {
                "accuracy": acc,
                "f1_score": f1,
                "correlation": corr
            }

    def _init_predictor(self):
        def f(dataset):
            if self.model is None:
                raise ValueError("Model has not been trained yet.")
            output = dataset.copy()
            features_col = [col for col in dataset.columns if col.startswith("mean_band_")]
            X = output[features_col].values
            output["predicted"] = self.model.predict(X)
            return output
        return f

    def _init_config(self):
        return {
            "name": "SVMModel",
            "description": "Predicts floods using a Support Vector Machine classifier.",
            "C": param_config(1.0, True, "float", 0.01, 100, log_scale=True),
            "kernel": param_config('rbf', True, "category", categories=['linear', 'poly', 'rbf', 'sigmoid']),
        }


class CSFIModel(Model):
        
    def _init_predictor(self):
        hardness = self.get("hardness")
        include_means = self.get("include_means")
        substract_means = self.get("substract_means")
        degree = self.get("degree")
        hidden_dim = self.get("hidden_dim")
        layers = self.get("layers")
        
        class CustomIndexPredictor(nn.Module):
            def __init__(self):
                super().__init__()
                self.hardness = hardness
                #Initialize the input size based on how we include means
                self.init_preprocess(include_means, substract_means, degree)
                #Initialize the fully connected layers
                self.nn = self.init_nnetwork(layers, hidden_dim)
                #Our model also learns a threshold s
                self.s = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=True)

            def forward(self, x):
                x = self.preprocess(x)
                x = self.nn(x)
                a,b = torch.abs(x).T
                denom = a + b
                index = torch.where(denom != 0, (a - b) / denom, torch.zeros_like(denom))
                comparison = index - self.s
                
                # The ouput is a continuous approximation of (a - b) / (a + b) > self.s
                return 1/(1 + torch.exp(-self.hardness*comparison)).unsqueeze(1)
            
            def init_preprocess(self, include_means, substract_means, degree):
                """
                Initialize the preprocessing parameters.
                """
                if include_means:
                    self.input_size = 8
                else :
                    self.input_size = 4
                if degree == 2:
                    self.input_size = (self.input_size + 1) * (self.input_size + 2) // 2
                def f(x):
                    """
                    Preprocess the input data.
                    """
                    if not isinstance(x, torch.Tensor):
                        x = torch.tensor(x, dtype=torch.float32)
                    if substract_means:
                        x[:, :4] = x[:, :4] - x[:, 4:]
                    if not include_means:
                        x = x[:, :4]
                    if degree == 2:
                        ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
                        x_aug = torch.cat([x, ones], dim=1)
                        # Compute the outer product for each sample in the batch
                        x_outer = torch.einsum('bi,bj->bij', x_aug, x_aug)
                        # Flatten only the upper triangle (including diagonal) for each sample
                        batch_size, dim, _ = x_outer.shape
                        triu_indices = torch.triu_indices(dim, dim)
                        x = x_outer[:, triu_indices[0], triu_indices[1]]
                    return x
                        
                self.preprocess = f
            
            def init_nnetwork(self, layers, hidden_dim):
                last_size = self.input_size
                layers_list = []
                for i in range(layers):
                    layers_list.append(nn.Linear(last_size, hidden_dim))
                    layers_list.append(nn.ReLU())
                    last_size = hidden_dim
                layers_list.append(nn.Linear(last_size, 2))  # Output layer with 2 outputs
                return nn.Sequential(*layers_list)
            
        self.model = CustomIndexPredictor()
        
        def f(dataset):
            """
            Predicts floods using the custom index model.
            """
            if self.model is None:
                raise ValueError("Model has not been trained yet.")
            self.model.eval()
            output = dataset.copy()
            feature_cols = [col for col in dataset.columns if col.startswith("mean_band_")]
            X = dataset[feature_cols].values.astype(np.float32)
            self.model.eval()
            with torch.no_grad():
                preds = self.model(X).squeeze().numpy()
                preds_binary = (preds > 0.5).astype(int)
            output["predicted"] = preds_binary
            output["proba"]= preds
            return output
        return f
    
    def _init_config(self):
        """
        Initialize the model configuration.
        """
        return {
            "name": "CSFIModel",
            "description": "Predicts floods using a custom index model based on mean band values.",
            "include_means": param_config(False, False, "bool", None, None),
            "substract_means": param_config(False, False, "bool", None, None),
            "hardness": param_config(1, True, "float", 0.1, 30, log_scale=True),
            "lr": param_config(0.01, True, "float", 1e-5, 1e-1, log_scale=True),
            "batch_size": param_config(16, True, "int", 1, 64),
            "epochs": param_config(20, True, "int", 1, 100),
            "degree": param_config(1, True, "int", 1, 2),
            "hidden_dim": param_config(16, True, "int", 8, 128),
            "layers": param_config(0, True, "int", 0, 3)
        }
 
    
    def train(self, test_size = 0.2):
        if self.dataset is None:
            raise ValueError("No dataset to train on. Please add data first.")

        # Prepare data
        self.dataset = self.dataset.dropna()
        
        feature_cols = [col for col in self.dataset.columns if col.startswith("mean_band_")]
        X = self.dataset[feature_cols].values.astype(np.float32)
        y = self.dataset["flooded"].values.astype(np.float32)
    
        X = torch.tensor(X)
        y = torch.tensor(y).unsqueeze(1)
        
        if test_size > 0:
            X, X_test, y, y_test = train_test_split(X, y, test_size=test_size)

        # Training parameters
        batch_size = self.get("batch_size")
        epochs = self.get("epochs")
        lr = self.get("lr")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        self.model.train()
        for epoch in tqdm(range(epochs), desc="Training Custom Index Model"):
            perm = torch.randperm(X.size(0))
            X_shuffled = X[perm]
            y_shuffled = y[perm]
            for i in range(0, X.size(0), batch_size):
                xb = X_shuffled[i:i+batch_size]
                yb = y_shuffled[i:i+batch_size]
                
                if xb.shape[0] > 0 :
                    output = self.model(xb)
                    loss = loss_fn(output, yb)
                    
                    assert not torch.isnan(loss), f"NaN loss encountered: {loss}"
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

        if test_size > 0:
            self.model.eval()
            with torch.no_grad():
                preds = self.model(X_test).squeeze().numpy()
                preds_binary = (preds > 0.5).astype(int)
                y_true = y_test.squeeze().numpy().astype(int)

            acc = accuracy_score(y_true, preds_binary)
            f1 = f1_score(y_true, preds_binary)
            correlation = matthews_corrcoef(y_true, preds_binary)
            
            self.training_results = {
                "accuracy": acc,
                "f1_score": f1,
                "correlation": correlation}
            
class LCSFIModel(CSFIModel):
    def _init_config(self):
        """
        Initialize the model configuration.
        """
        return {
            "name": "LCSFIModel",
            "description": "Predicts floods using a custom index model based on mean band values.",
            "include_means": param_config(False, False, "bool", None, None),
            "substract_means": param_config(False, False, "bool", None, None),
            "hardness": param_config(1, True, "float", 0.1, 30, log_scale=True),
            "lr": param_config(0.01, True, "float", 1e-5, 1e-1, log_scale=True),
            "batch_size": param_config(16, True, "int", 1, 64),
            "epochs": param_config(20, True, "int", 1, 100),
            "degree": param_config(1, True, "int", 1, 1),
            "hidden_dim": param_config(16, True, "int", 8, 128),
            "layers": param_config(0, True, "int", 0, 0)
        }
    
class SimpleNNModel(Model):
    def _init_predictor(self):
        include_means = self.get("include_means")
        substract_means = self.get("substract_means")
        hidden_dim = self.get("hidden_dim")
        layers = self.get("layers")
        
        class SimpleNN(nn.Module):
            def __init__(self):
                super().__init__()
                #Initialize the input size based on how we include means
                self.init_preprocess(include_means, substract_means)
                #Initialize the fully connected layers
                self.nn = self.init_nnetwork(layers, hidden_dim)

            def forward(self, x):
                x = self.preprocess(x)
                x = self.nn(x)
                x= torch.sigmoid(x)  # Apply sigmoid activation to output
                return x
            
            def init_preprocess(self, include_means, substract_means):
                """
                Initialize the preprocessing parameters.
                """
                if include_means:
                    self.input_size = 8
                else :
                    self.input_size = 4
                def f(x):
                    """
                    Preprocess the input data.
                    """
                    if not isinstance(x, torch.Tensor):
                        x = torch.tensor(x, dtype=torch.float32)
                    if substract_means:
                        x[:, :4] = x[:, :4] - x[:, 4:]
                    if not include_means:
                        x = x[:, :4]
                    return x
                        
                self.preprocess = f
            
            def init_nnetwork(self, layers, hidden_dim):
                last_size = self.input_size
                layers_list = []
                for i in range(layers):
                    layers_list.append(nn.Linear(last_size, hidden_dim))
                    layers_list.append(nn.ReLU())
                    last_size = hidden_dim
                layers_list.append(nn.Linear(last_size, 1))  # Output layer with 2 outputs
                return nn.Sequential(*layers_list)

        self.model = SimpleNN()

        def f(dataset):
            if self.model is None:
                raise ValueError("Model has not been trained yet.")
            self.model.eval()
            output = dataset.copy()
            feature_cols = [col for col in dataset.columns if col.startswith("mean_band_")]
            X = dataset[feature_cols].values.astype(np.float32)
            with torch.no_grad():
                preds = self.model(X).squeeze().numpy()
                preds_binary = (preds > 0.5).astype(int)
            output["predicted"] = preds_binary
            return output
        return f

    def _init_config(self):
        return {
            "name": "SimpleNNModel",
            "description": "Predicts floods using a simple feedforward neural network.",
            "hidden_dim": param_config(16, True, "int", 4, 64),
            "layers": param_config(2, True, "int", 1, 3),
            "epochs": param_config(20, True, "int", 1, 100),
            "lr": param_config(1e-3, True, "float", 1e-5, 1e-1, log_scale=True),
            "batch_size": param_config(16, True, "int", 1, 64),
            "include_means": param_config(False, True, "bool", None, None),
            "substract_means": param_config(True, True, "bool", None, None),
        }

    def train(self, test_size=0.2):
        if self.dataset is None:
            raise ValueError("No dataset to train on. Please add data first.")

        self.dataset = self.dataset.dropna()
        feature_cols = [col for col in self.dataset.columns if col.startswith("mean_band_")]
        X = self.dataset[feature_cols].values.astype(np.float32)
        y = self.dataset["flooded"].values.astype(np.float32)

        X = torch.tensor(X)
        y = torch.tensor(y).unsqueeze(1)

        if test_size > 0:
            X, X_test, y, y_test = train_test_split(X, y, test_size=test_size)

        epochs = self.get("epochs")
        batch_size = self.get("batch_size")
        lr = self.get("lr")


        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        self.model.train()
        for epoch in tqdm(range(epochs), desc="Training SimpleNNModel"):
            perm = torch.randperm(X.size(0))
            X_shuffled = X[perm]
            y_shuffled = y[perm]
            for i in range(0, X.size(0), batch_size):
                xb = X_shuffled[i:i+batch_size]
                yb = y_shuffled[i:i+batch_size]
                output = self.model(xb)
                loss = loss_fn(output, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if test_size > 0:
            self.model.eval()
            with torch.no_grad():
                preds = self.model(X_test).squeeze().numpy()
                preds_binary = (preds > 0.5).astype(int)
                y_true = y_test.squeeze().numpy().astype(int)
            acc = accuracy_score(y_true, preds_binary)
            f1 = f1_score(y_true, preds_binary)
            correlation = matthews_corrcoef(y_true, preds_binary)
            self.training_results = {
                "accuracy": acc,
                "f1_score": f1,
                "correlation": correlation
            }
            
def param_config(value, trainable, param_type, min = None, max= None, log_scale=False, categories =None):
    """
    Helper function to create a parameter configuration dictionary.
    """
    return {
        "value": value,
        "trainable": trainable,
        "type": param_type,
        "log_scale": log_scale,
        "min": min,
        "max": max,
        "categories": categories
    }
    
def param_config_to_suggestion(name, param_config, trial):
    if param_config["type"] == "float":
        return trial.suggest_float(name, param_config["min"], param_config["max"], log=param_config.get("log", False))
    elif param_config["type"] == "int":
        return trial.suggest_int(name, param_config["min"], param_config["max"])
    elif param_config["type"] == "bool":
        return trial.suggest_categorical(name, [True, False])
    elif param_config["type"] == "category":
        return trial.suggest_categorical(name, param_config["categories"])
    else:
        raise ValueError(f"Unsupported parameter type: {param_config['type']}")

def get_model_class(model_name, config = None, **kwargs):
    """
    Get the model class based on the model name.
    """
    if model_name == "CSFIModel":
        model_instance = CSFIModel(config = config, **kwargs)
    elif model_name == "LCSFIModel":
        model_instance = LCSFIModel(config = config, **kwargs)
    elif model_name == "XGBoostModel":
        model_instance = XGBoostModel(config = config, **kwargs)
    elif model_name == "SimpleNNModel":
        model_instance = SimpleNNModel(config = config, **kwargs)
    elif model_name == "RandomForestModel":
        model_instance = RandomForestModel(config = config, **kwargs)
    elif model_name == "SVMModel":
        model_instance = SVMModel(config = config, **kwargs)
    elif model_name == "NDWIMeanModel":
        model_instance = NDWIMeanModel(config = config, **kwargs)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
        
    return model_instance

def save_model(model_instance, path, name = None):
    if not name:
        name = model_instance.model_config["name"]
    model_dir = os.path.join(path, name)
    os.makedirs(model_dir, exist_ok=True)
    
    config = model_instance.model_config
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    if name in ["CSFIModel", "SimpleNNModel", "LCSFIModel"]:
        torch.save(model_instance.model.state_dict(), os.path.join(model_dir, "weights.pth"))
    elif name == "XGBoostModel":
        model_instance.model.save_model(os.path.join(model_dir, "weights.json"))
    elif name in ["RandomForestModel", "SVMModel"]:
        import joblib
        joblib.dump(model_instance.model, os.path.join(model_dir, "weights.joblib"))
    else:
        raise ValueError(f"Unsupported model name for saving: {name}")

def load_model(path):
    """
    Load a model instance from the specified path.
    """
  
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model directory {path} does not exist.")

    with open(os.path.join(path, "config.json"), "r") as f:
        config = json.load(f)
    model = config["name"]
    model_class = get_model_class(model, config=config)
    
    if model in ["CSFIModel", "SimpleNNModel", "LCSFIModel"]:
        model_class.model.load_state_dict(torch.load(os.path.join(path, "weights.pth"), weights_only=True))
    elif model == "XGBoostModel":
        model_class.model.load_model(os.path.join(path, "weights.json"))
    elif model in ["RandomForestModel", "SVMModel"]:
        import joblib
        model_class.model = joblib.load(os.path.join(path, "weights.joblib"))
        
    return model_class