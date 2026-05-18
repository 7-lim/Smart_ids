"""
AI-SmartTIDS — intelligent real-time Intrusion Detection System.

Public surface for downstream teams:

    from src import SmartTIDS_Predictor

Internal modules (`config`, `preprocessing`) support the predictor.
Optional modules: `api` (HTTP service) and `flow_extractor` (pcap → flow).
Model-architecture and metric helpers used only at training time live
directly in the notebooks.
"""
from src.inference import SmartTIDS_Predictor, PredictionResult

__all__ = ["SmartTIDS_Predictor", "PredictionResult"]
__version__ = "1.0.0"
