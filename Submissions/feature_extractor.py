import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

non_numerical = list(['A6', 'A 6', 'V .', 'M 0', 'A5', 'N 0', 'M 5', 'V 0', 'D 5', 'M', 'N 1',
                     'V 1', 'M 7', 'S 6', 'A 7', 'A 1', '5', 'M 1', 'D 7', 'A 0', 'M 6', 'A 3',
                     'N', 'A 5', 'A 8', 'D 6', '6', 'A 4', 'M5', 'M6', 'A4', 'A 9'])
map_dict = {'VOLKSWAGEN': 'VW',
                'VOLVO': 'VOLVO CAR', # 'VOLVO BUS', 'VOLVO TRUCK'
                'MG': 'MG MOTOR',     # 'MG ROVER'
                'CHEVROLET': 'CHEVROLET UK', # 'CHEVROLET USA'
                'LEXUS': 'TOYOTA',
                'QUATTRO': 'AUDI',
                'MERCEDES AMG': 'MERCEDES BENZ', # 'MERCEDES BENZ CARS UK LTD', 'MERCEDES BENZ VANS UK LTD', 'MERCEDES BENZ BUS', 'MERCEDES BENZ TRUCKS UK LTD'
                'LTI VEHICLES': 'LTI',
                'DANGEL': 'DANGEL',
                'THE LONDON TAXI COMPANY': 'LTI',
                'RENAULT TECH': 'RENAULT', # 'RENAULT TRUCKS UK LTD', 'RENAULTTRUCKS UK LTD', 'RENAULT VI', 'RENAULT AGRICULTURE'
                'JAGUAR LAND ROVER LIMITED': 'JAGUAR',
                'MERCEDES': 'MERCEDES BENZ', # 'MERCEDES BENZ CARS UK LTD', 'MERCEDES BENZ VANS UK LTD', 'MERCEDES BENZ BUS', 'MERCEDES BENZ TRUCKS UK LTD'
                'BMW I': 'BMW', # 'BMW MOTORCYCLES', 'BMW MOTORRAD'
                'FORD CNG TECHNIK': 'FORD',
               }


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        path = os.path.dirname(__file__)
        df_co2 = pd.read_csv(os.path.join(path,'data_co2.csv'), low_memory=False)
        
        
        df_co2['Marque'] = df_co2['Marque'].apply(lambda x : x.strip())
        df_co2['Marque'] = df_co2['Marque'].apply(lambda x : x.replace('-', ' '))

        for col in ['Consommation mixte (l/100km)', 'Consommation urbaine (l/100km)', 'Puissance maximale (kW)']:
            df_co2[col] = df_co2[col].apply(
                lambda x: str(x).replace(',','.')).astype(float)

        for col in ['Consommation extra-urbaine (l/100km)', 'Boîte de vitesse']:
            df_co2[col] = df_co2[col].apply(
                lambda x: str(x).replace(',','.'))
            df_co2[col] = df_co2[col].apply(
                lambda x: str(x).strip())
        
        for index, row in tqdm(df_co2[~(df_co2['Boîte de vitesse'].isin(non_numerical))].iterrows()):
            switch_value = row['Boîte de vitesse']
            df_co2.loc[index,'Boîte de vitesse'] = row['Consommation extra-urbaine (l/100km)']
            df_co2.loc[index,'Consommation extra-urbaine (l/100km)'] = switch_value
        
        df_co2 = df_co2[~df_co2['Consommation extra-urbaine (l/100km)'].isin(non_numerical)]
        df_co2['Consommation extra-urbaine (l/100km)'] = df_co2['Consommation extra-urbaine (l/100km)'].astype(float)
        df_co2.drop(['Unnamed: 0', 'CNIT'], axis=1, inplace=True)
        
        
        
        df_co2.Marque = df_co2.Marque.replace(map_dict)
        df_co2.Marque = df_co2.Marque.str.lower()

        self.df_co2 = df_co2

        return self

    def transform(self, X_df):
        X_encoded = X_df.copy()
        
        
        path = os.path.dirname(__file__)
        df_groupe_marque = pd.read_excel(os.path.join(path,'voiture_groupe_marque.xlsx'))
        
        
        

        marque_found = []
        marque_not_found = []
        for a in self.df_co2.Marque.unique():
            if a in X_encoded.Make.unique():
                marque_found.append(a)
            else:
                marque_not_found.append(a)
        
        modele_found = []
        modele_not_found = []

        for a in self.df_co2['Modèle dossier'].unique():
            if a in X_encoded.Model.unique():
                modele_found.append(a)
            else:
                modele_not_found.append(a)
        df_co2_num = self.df_co2.groupby(['Marque','Modèle dossier', 'Annee']).mean().reset_index()

        df_co2_non_num = self.df_co2.groupby(['Marque','Modèle dossier', 'Annee'])['Boîte de vitesse'
                                                                        ].agg(pd.Series.mode).reset_index()
        self.df_co2 = df_co2_num.merge(df_co2_non_num, on=['Marque','Modèle dossier', 'Annee'])
        
        
        
        X_encoded['Make'] = X_encoded['Make'].apply(lambda x : x.strip())
        X_encoded['Make'] = X_encoded['Make'].apply(lambda x : x.replace('-', ' '))
        df_groupe_marque['Marques'] = df_groupe_marque['Marques'].str.upper()

        X_encoded['Launch Date']= pd.to_datetime(X_encoded['Launch Date'])
        X_encoded['Launch Year']= X_encoded['Launch Date'].dt.year

        X_encoded['Build Start']= pd.to_datetime(X_encoded['Build Start'], errors= 'coerce')
        X_encoded['Build End']= pd.to_datetime(X_encoded['Build End'], errors= 'coerce')

        X_encoded.Model = X_encoded.Model.fillna(value=X_encoded['Recalls Model Information'])
        
        print(int(len(X_encoded[X_encoded.Make.isin(marque_found)])*100/len(X_encoded)),
              '% de données de recall restantes après merge sur marque avec CO2')
        print(int(len(X_encoded[X_encoded.Model.isin(modele_found)])*100/len(X_encoded)),
             '% de données de recall restantes après merge sur modèle avec CO2')
        print(int(len(X_encoded[X_encoded.Make.isin(marque_found)][X_encoded.Model.isin(modele_found)])*100/len(X_encoded)),
              '% de données de recall restantes après merge sur marque et modèle avec CO2')
        
        
        condition = (X_encoded.Make.isin(marque_found)) & (X_encoded.Model.isin(modele_found))
        num_cols = ['Consommation mixte (l/100km)', 'Consommation urbaine (l/100km)', 'CO2 (g/km)',
                    'Consommation extra-urbaine (l/100km)', 'Puissance maximale (kW)']

        for col in num_cols:
            X_encoded[col] = float("NaN")

        for index, row in tqdm(X_encoded[condition].iterrows()):
            make = row['Make']
            model = row['Model']
            year = row['Launch Year']
            condition_match = (self.df_co2['Marque'] == make) & (self.df_co2['Modèle dossier'] == model)
            after_index = []
            before_index = []
            for index_2, row_2 in self.df_co2[condition_match].iterrows():
                if row_2['Annee'] >= year:
                    after_index.append(index_2)
                elif row_2['Annee'] < year:
                    before_index.append(index_2)
            if after_index:
                X_encoded.loc[index, num_cols] = self.df_co2.loc[after_index, num_cols].mean()
            elif before_index:
                X_encoded.loc[index, num_cols] = self.df_co2.loc[before_index, num_cols].mean()
        
        keep_cols = num_cols + ['Launch Year', 'Vehicle Numbers']
        
        X_encoded = X_encoded[keep_cols]
        
        numeric_transformer = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='median'))
        ])

        preprocessor_comp = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, keep_cols),
            ])
        
        X_array = preprocessor_comp.fit_transform(X_encoded)
        
        
        return X_array