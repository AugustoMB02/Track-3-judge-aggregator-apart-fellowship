#!/usr/bin/env python3
"""
MLP-based Iterative Judge Selection with GPU Acceleration

This module extends the base IterativeJudgeSelector to use MLP aggregators
with GPU acceleration and gradient-based attribution for judge importance.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.core.aggregator_training import MLPTrainer, compute_metrics
from experiments.track3_automated_selection.iterative_selection.iterative_selection import (
    IterativeJudgeSelector,
    SelectionConfig,
    IterationResult,
)
from experiments.track3_automated_selection.iterative_selection.judge_set_metrics import (
    JudgeSetEvaluator,
)
from experiments.track3_automated_selection.iterative_selection.gap_analyzer import (
    GapAnalyzer,
    identify_least_important_judge,
)

# Import Track 2 gradient-based attribution
from experiments.track2_judge_interpretability.explainability.fetch_attributions import (
    compute_input_x_gradient_batch,
)

logger = logging.getLogger(__name__)


class MLPJudgeSelector(IterativeJudgeSelector):
    """GPU-accelerated judge selector using MLP aggregators."""
    
    def __init__(
        self,
        config: SelectionConfig,
        device: str = "cuda",
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        n_epochs: int = 100,
        dropout: float = 0.2,
        l2_reg: float = 0.001,
        early_stopping_patience: int = 15,
    ):
        """
        Initialize MLP-based selector with GPU support.
        
        Args:
            config: Selection configuration
            device: Device to use ('cuda' or 'cpu')
            hidden_dim: Hidden layer dimension for MLP
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            n_epochs: Maximum training epochs
            dropout: Dropout probability
            l2_reg: L2 regularization strength
            early_stopping_patience: Patience for early stopping
        """
        super().__init__(config)
        
        # MLP configuration
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.early_stopping_patience = early_stopping_patience
        
        logger.info(f"Initialized MLPJudgeSelector on device: {self.device}")
        logger.info(f"GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    def _train_aggregator(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        judge_names: List[str],
    ) -> MLPTrainer:
        """
        Train MLP aggregator on GPU.
        
        Args:
            X_train: Training judge scores
            y_train: Training targets
            judge_names: Names of judges (for logging)
            
        Returns:
            Trained MLPTrainer instance
        """
        # Create validation split from training data
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config.validation_split,
            random_state=42,
        )
        
        # Initialize trainer
        mlp = MLPTrainer(
            hidden_dim=self.hidden_dim,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            dropout=self.dropout,
            l2_reg=self.l2_reg,
            early_stopping_patience=self.early_stopping_patience,
            device=str(self.device),
        )
        
        # Train on GPU
        logger.info(f"Training MLP with {len(judge_names)} judges on {self.device}")
        train_losses, val_losses = mlp.fit(X_train_sub, y_train_sub, X_val, y_val)
        
        return mlp
    
    def _compute_gradient_importance(
        self,
        mlp: MLPTrainer,
        X: np.ndarray,
        judge_names: List[str],
    ) -> Dict[str, float]:
        """
        Compute judge importance using gradient-based attribution.
        
        Uses Input × Gradient method from Track 2 to compute how much each
        judge contributes to the final prediction.
        
        Args:
            mlp: Trained MLP model
            X: Input judge scores (n_samples, n_judges)
            judge_names: Names of judges
            
        Returns:
            Dictionary mapping judge names to importance scores
        """
        mlp.model.eval()
        mlp.model.zero_grad()
        
        # Convert to tensor on GPU
        X_tensor = torch.FloatTensor(X).to(self.device)
        X_tensor.requires_grad = True
        
        # Forward pass
        outputs = mlp.model(X_tensor)
        
        # Backward pass to compute gradients
        # Sum all outputs to get scalar for backward (we want gradient w.r.t. all predictions)
        outputs.sum().backward()
        
        # Compute Input × Gradient attribution
        with torch.no_grad():
            attributions = (X_tensor * X_tensor.grad).cpu().numpy()
        
        # Aggregate attributions across samples
        # Mean absolute attribution per judge
        importance = np.abs(attributions).mean(axis=0)
        
        # Normalize to [0, 1] range
        if importance.max() > 0:
            importance = importance / importance.max()
        
        # Map to judge names
        importance_dict = {
            name: float(score)
            for name, score in zip(judge_names, importance)
        }
        
        return importance_dict
    
    def _compute_variance_importance(
        self,
        mlp: MLPTrainer,
        X: np.ndarray,
        judge_names: List[str],
    ) -> Dict[str, float]:
        """
        Compute importance based on variance of attributions.
        
        This captures judges that are critical for specific subsets of data.
        
        Args:
            mlp: Trained MLP model
            X: Input judge scores
            judge_names: Names of judges
            
        Returns:
            Dictionary mapping judge names to variance-based importance
        """
        mlp.model.eval()
        
        # Compute per-sample attributions
        attributions_per_sample = []
        
        for i in range(len(X)):
            X_sample = torch.FloatTensor(X[i:i+1]).to(self.device)
            X_sample.requires_grad = True
            
            output = mlp.model(X_sample)
            output.backward()
            
            attribution = (X_sample * X_sample.grad).detach().cpu().numpy()[0]
            attributions_per_sample.append(attribution)
            
            mlp.model.zero_grad()
        
        # Compute variance across samples for each judge
        attributions_array = np.array(attributions_per_sample)
        variance_importance = np.var(np.abs(attributions_array), axis=0)
        
        # Normalize
        if variance_importance.max() > 0:
            variance_importance = variance_importance / variance_importance.max()
        
        importance_dict = {
            name: float(score)
            for name, score in zip(judge_names, variance_importance)
        }
        
        return importance_dict
    
    def _evaluate_iteration(
        self,
        iteration: int,
        judge_names: List[str],
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        removed_judge: Optional[str] = None,
        added_judge: Optional[str] = None,
    ) -> IterationResult:
        """
        Run a single iteration with MLP aggregator and gradient-based importance.
        
        Returns:
            IterationResult with all metrics and analysis
        """
        # Train MLP aggregator on GPU
        mlp = self._train_aggregator(X_train, y_train, judge_names)
        
        # Get predictions
        train_predictions = mlp.predict(X_train)
        test_predictions = mlp.predict(X_test)
        
        # Compute regression metrics
        train_metrics = compute_metrics(y_train, train_predictions)
        test_metrics = compute_metrics(y_test, test_predictions)
        
        # Compute importance using gradient-based attribution
        logger.info("Computing gradient-based importance...")
        grad_importance = self._compute_gradient_importance(mlp, X_test, judge_names)
        
        # Compute variance-based importance (specialist judges)
        logger.info("Computing variance-based importance...")
        var_importance = self._compute_variance_importance(mlp, X_test, judge_names)
        
        # Combine both importance metrics (50/50 weight)
        combined_importance = {}
        for name in judge_names:
            combined_importance[name] = 0.5 * grad_importance[name] + 0.5 * var_importance[name]
        
        # Evaluate judge set
        judge_set_metrics = self.judge_set_evaluator.evaluate(
            judge_scores=X_test,
            judge_names=judge_names,
            predictions=test_predictions,
            targets=y_test,
            importance_scores=combined_importance,
        )
        
        # Gap analysis
        gap_result = self.gap_analyzer.analyze(
            predictions=test_predictions,
            targets=y_test,
            judge_scores=X_test,
            judge_names=judge_names,
        )
        
        # Calculate improvement
        current_r2 = test_metrics.get("r2", 0.0)
        improvement = current_r2 - self.best_r2
        
    def _check_stopping_criteria(
        self,
        iteration: int,
        n_judges: int,
        current_r2: float,
        improvement: float,
    ) -> Tuple[bool, Optional[str]]:
        """Check if selection should stop."""
        
        # Max iterations reached
        if iteration >= self.config.max_iterations:
            return True, "max_iterations_reached"
        
        # Target number of judges reached (main criterion)
        if n_judges <= self.config.target_judges:
            return True, f"target_judges_reached_{self.config.target_judges}"
        
        # Absolute minimum judges reached (safety)
        if n_judges <= self.config.min_judges:
            return True, "min_judges_safety_limit"
        
        # Performance degradation too severe
        if improvement < -self.config.r2_degradation_threshold:
            return True, f"performance_degraded_by_{abs(improvement):.4f}"
        
        # Plateau detected
        if self.plateau_count >= self.config.plateau_patience:
            return True, f"plateau_detected_after_{self.plateau_count}_iterations"
        
        return False, None
        
        # Save model checkpoint
        if self.config.save_intermediate:
            model_path = self.output_dir / f"iteration_{iteration:02d}" / "mlp_model.pt"
            model_path.parent.mkdir(exist_ok=True, parents=True)
            mlp.save_model(model_path)
        
        result = IterationResult(
            iteration=iteration,
            judge_names=judge_names,
            n_judges=len(judge_names),
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            judge_set_metrics=judge_set_metrics.to_dict(),
            importance_scores=combined_importance,
            removed_judge=removed_judge,
            added_judge=added_judge,
            gap_analysis=gap_result.to_dict(),
            improvement=improvement,
            should_stop=should_stop,
            stop_reason=stop_reason,
        )
        
        return result
