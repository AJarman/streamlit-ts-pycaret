from typing import Callable, Dict

from sktime.datasets import load_airline, load_lynx, load_shampoo_sales

DATASET_MAP: Dict[str, Callable] = {
    "Airline": load_airline,
    "Lynx": load_lynx,
    "Shampoo": load_shampoo_sales
}


# def load_datasets(name: str) -> pd.DataFrame:
