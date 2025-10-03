from typing import Union, Dict

def compute_mix(dose_per_litre: Union[float, int], tank_l: float, area_ha: float, spray_vol: float) -> Dict[str, float]:
    """Compute mixing dosage plan and return a structured dict."""
    total_water = spray_vol * area_ha  # liters
    total_pesticide = dose_per_litre * total_water  # ml (assuming dose_per_litre in ml/L)
    tanks = total_water / tank_l if tank_l > 0 else 0
    per_tank_water = tank_l
    per_tank_pesticide = dose_per_litre * tank_l
    return {
        'total_water_l': round(total_water, 2),
        'total_pesticide_ml': round(total_pesticide, 2),
        'tanks': round(tanks, 1),
        'per_tank_water_l': round(per_tank_water, 2),
        'per_tank_pesticide_ml': round(per_tank_pesticide, 2),
        'dose_per_litre': dose_per_litre,
        'units_label': 'ml'
    }
