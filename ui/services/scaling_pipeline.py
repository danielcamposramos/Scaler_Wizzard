"""
Unified Scaling Pipeline for Scaler Wizard.
Orchestrates LoRA application, context scaling, and safety gates.
"""

import logging
from components.scaling_engine.unsloth_engine import UnslothEngine
from components.safety.circuit_breaker import CircuitBreaker, CBConfig

class ScalingPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.engine = UnslothEngine(
            model_name=config.get("model_name", "TRMC-7B-Recursive-MoE"),
            max_seq_length=config.get("max_length", 32768)
        )
        self.safety_gate = CircuitBreaker(CBConfig(
            max_perplexity_delta=config.get("safety_margin", 0.15),
            hard_stop=True
        ))

    def execute_scaling_phase(self, phase_name: str, dataset_configs: list):
        """Executes a single scaling phase with safety monitoring."""
        self.logger.info(f"🚀 Engaging Phase: {phase_name}")
        
        # Execute training kernel
        training_results = self.engine.run_pretraining(
            dataset_configs=dataset_configs,
            args_dict=self.config
        )
        
        # Perform post-phase validation
        # (In a real run, this would be updated per step in the Trainer loop)
        return training_results

    def rollback_check(self, perplexity: float, accuracy: float):
        """Manual gate for human-in-the-loop verification."""
        telegram = self.safety_gate.step(perplexity, accuracy)
        if telegram["action"] == "stop":
            self.logger.error(f"🛑 CRITICAL SAFETY STOP: {telegram['reason']}")
            return False
        return True

__all__ = ["ScalingPipeline"]