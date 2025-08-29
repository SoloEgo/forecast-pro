from dataclasses import dataclass
@dataclass
class Config:
    horizon: int = 12
    id_col: str = 'sku'
    date_col: str = 'date'
    y_col: str = 'qty'
    promo_col: str = 'promo_flag'
