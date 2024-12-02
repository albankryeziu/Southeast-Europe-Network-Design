import numpy as np
import pandas as pd

import pandas as pd


def data():
    """Function that returns the dictionaries and lists of the data used for this project"""
    return{
        'sources_romania': [
    {"name": "RO1 SC ROMPETROL RAFINARE SA", "amount_1": 993000, "type": "source", "lon": 28.646779, "lat": 44.339117},
    {"name": "RO2 LIBERTY GALATI SA", "amount_1": 2000000, "type": "source", "lon": 27.972194, "lat": 45.440355},
    {"name": "RO3 S.C. CHEMGAS HOLDING CORPORATION SRL", "amount_1": 223000, "type": "source", "lon": 27.385171, "lat": 44.533743},
    {"name": "RO4 HEIDELBERGCEMENT ROMANIA SA", "amount_1": 731000, "type": "source", "lon": 26.029587, "lat": 46.897861},
    {"name": "RO5 SC HOLCIM ROMANIA SA", "amount_1": 1040000, "type": "source", "lon": 25.1110277, "lat": 45.2901138},
    {"name": "RO6 CRH CIMENT (ROMANIA)S.A.", "amount_1": 876000, "type": "source", "lon": 28.307913, "lat": 44.241846},
    {"name": "RO7 SC PETROTEL LUKOIL SA", "amount_1": 432000, "type": "source", "lon": 26.079463, "lat": 44.946413},
    {"name": "RO8 HOLCIM (Romania) SA - Ciment Alesd", "amount_1": 940000, "type": "source", "lon": 22.331876, "lat": 47.038682},
    {"name": "RO9 OMV PETROM SA - Petrobrazi", "amount_1": 984000, "type": "source", "lon": 26.010388, "lat": 44.876519},
    {"name": "RO10 SC CRH CIMENT (ROMANIA) SA", "amount_1": 747000, "type": "source", "lon": 25.281858, "lat": 45.951652},
    {"name": "RO11 HEIDELBERGCEMENT ROMANIA SA - Fieni", "amount_1": 716000, "type": "source", "lon": 25.41758611, "lat": 45.12266666},
    {"name": "RO12 FABRICA DE CIMENT CHISCADAGA", "amount_1": 664000, "type": "source", "lon": 22.866436, "lat": 45.954952},
    {"name": "RO13 SC AZOMURES SA", "amount_1": 1460000, "type": "source", "lon": 24.506814, "lat": 46.515151},
],
        'sources_bulgaria':[],
        
    }

def parameters():
    """Function that is used to get the parameters such as costs, enviroment and so on"""
    return {
        'transportation_costs':[1,2,3]
    }


def load_emitter_data(file_path: str, sheet_name: str = None):
    """
    Load and process emitter data from an Excel sheet.

    Parameters:
        file_path (str): The path to the Excel file.
        sheet_name (str, optional): The name of the sheet to read. Defaults to the first sheet.

    Returns:
        pd.DataFrame: Processed DataFrame containing emitter data.
    """
    # Load data from the specified Excel sheet
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # Drop unnamed columns (columns 8 and 9 in zero-based indexing)
    df = df.drop(df.columns[[8, 9]], axis=1)

    return df

file_path= r"C:\Users\Alban\OneDrive - University of Groningen\Desktop\research\Master thesis\daniel\data\sources.xlsx"
emitter_data=load_emitter_data(file_path, "emitters")
#emitter_data=pd.read_excel(file_path,"emitters")
#emitter_data=emitter_data.drop(emitter_data.columns[[8,9]],axis=1)
third_row=emitter_data.iloc[2].to_dict()
print(emitter_data.columns)
print(third_row)

print(emitter_data)
