import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class VisionResult:
    label: str
    confidence: float

@dataclass
class SeverityResult:
    percent: float

class AgentOrchestrator:
    def __init__(self, kb_path: str = 'data/knowledge_base.csv'):
        self.kb = pd.read_csv(kb_path)

    def decide(self, vision: Optional[VisionResult], severity: SeverityResult, crop: str = 'generic') -> Dict[str, Any]:
        if vision is None or vision.confidence < 0.55:
            inferred = 'leaf_spot' if severity.percent > 12 else 'healthy'
            reason = 'Vision uncertain; fallback to color heuristic.'
            conf = 0.5
        else:
            inferred = vision.label
            reason = 'Vision model confident.'
            conf = vision.confidence

        stage = 'late' if severity.percent/100.0 >= 0.20 else ('early' if severity.percent/100.0 >= 0.08 else 'any')

        rec = self._lookup(inferred, stage, severity.percent/100.0)
        action = {
            'diagnosis': inferred,
            'confidence': round(conf,3),
            'stage': stage,
            'severity_percent': round(severity.percent,1),
            'recommendation': rec,
            'rationale': reason
        }
        if inferred == 'healthy':
            action['safety'] = 'No spray recommended. Re-check in 3-5 days.'
        else:
            action['safety'] = 'Wear PPE and follow label instructions. Rotate FRAC codes.'
        return action

    def _lookup(self, disease: str, stage: str, sev: float) -> Dict[str, Any]:
        subset = self.kb[self.kb['disease'].eq(disease)]
        if subset.empty:
            subset = self.kb[self.kb['disease'].eq('leaf_spot')]
        stage_rows = subset[subset['stage'].eq(stage)]
        if stage_rows.empty:
            stage_rows = subset[subset['stage'].eq('any')]
        stage_rows = stage_rows.sort_values('severity_threshold')
        chosen = None
        for _, row in stage_rows.iterrows():
            if sev >= float(row['severity_threshold']):
                chosen = row
        if chosen is None:
            chosen = stage_rows.iloc[0]
        return {
            'pesticide': chosen['pesticide_ai_name'],
            'frac_code': chosen['frac_code'],
            'mode_of_action': chosen['mode_of_action'],
            'dose_per_litre': float(chosen['dose_per_litre']),
            'formulation': chosen['formulation'],
            'phi_days': chosen['preharvest_interval'],
            'notes': chosen['notes']
        }
