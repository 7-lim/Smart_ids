"""
AI-SmartTIDS — intelligent real-time Intrusion Detection System.

Public surface for downstream teams:

    from src import SmartTIDS_Predictor

Internal modules (`config`, `preprocessing`, `models`, `gan_balancer`,
`evaluation`) are also importable for the AI/ML team.
"""
from src.inference import SmartTIDS_Predictor, PredictionResult

__all__ = ["SmartTIDS_Predictor", "PredictionResult"]
__version__ = "1.0.0"
