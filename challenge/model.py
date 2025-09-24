import pandas as pd

from typing import Tuple, Union, List

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # Compute delay if requested. The raw dataset doesn't include it.
        if target_column == "delay":
            fecha_o = pd.to_datetime(data["Fecha-O"], format="%Y-%m-%d %H:%M:%S")
            fecha_i = pd.to_datetime(data["Fecha-I"], format="%Y-%m-%d %H:%M:%S")
            min_diff = (fecha_o - fecha_i).dt.total_seconds() / 60.0
            target = (min_diff > 15).astype(int).to_frame(name="delay")
        else:
            target = None

        # Build features from OPERA, TIPOVUELO and MES and align to the fixed set
        opera_d = pd.get_dummies(data["OPERA"], prefix="OPERA")
        tipo_d = pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO")
        mes_d = pd.get_dummies(data["MES"], prefix="MES")
        features_all = pd.concat([opera_d, tipo_d, mes_d], axis=1)

        required_columns = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air",
        ]
        features = features_all.reindex(columns=required_columns, fill_value=0)

        if target_column == "delay":
            return features, target
        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Compute weights to emphasize minority class and fit a simple LR
        from sklearn.linear_model import LogisticRegression

        y = target.iloc[:, 0]
        n_y0 = int((y == 0).sum())
        n_y1 = int((y == 1).sum())
        class_weight = {1: n_y0 / len(y), 0: n_y1 / len(y)}

        model = LogisticRegression(class_weight=class_weight, random_state=1)
        model.fit(features, y)
        self._model = model

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            return [0 for _ in range(features.shape[0])]
        preds = self._model.predict(features)
        return [int(v) for v in preds]