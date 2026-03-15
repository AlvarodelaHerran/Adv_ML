import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class IQRCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        # Permite ajustar el multiplicador del IQR si quisieras optimizarlo luego
        self.factor = factor
        self.limits_ = {}
        self.cols_to_cap_ = []

    def fit(self, X, y=None):
        # Aquí calculamos y guardamos los límites (SOLO se ejecuta en train)
        X_df = pd.DataFrame(X) # Por si entra como array de numpy
        self.cols_to_cap_ = X_df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in self.cols_to_cap_:
            Q1 = X_df[col].quantile(0.25)
            Q3 = X_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.factor * IQR
            upper = Q3 + self.factor * IQR
            
            # Guardamos los límites en el diccionario interno
            self.limits_[col] = (lower, upper)
            
        return self # Un fit siempre debe retornar self

    def transform(self, X):
        # Aquí aplicamos el capping con los límites ya calculados
        X_df = pd.DataFrame(X).copy() # Copia para no alterar el original
        
        for col in self.cols_to_cap_:
            if col in self.limits_:
                lower, upper = self.limits_[col]
                X_df[col] = X_df[col].clip(lower=lower, upper=upper)
                
        return X_df