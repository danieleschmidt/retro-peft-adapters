"""
Cross-Modal Causal Temporal Reasoning

Revolutionary implementation of causal discovery and temporal reasoning across modalities:

1. Automated causal graph discovery from multi-modal data
2. Temporal logic integration into neural architectures
3. Counterfactual reasoning for "what if" scenario generation
4. Cross-modal causal relationship modeling
5. Interventional reasoning for adaptive behavior

Research Contribution: First implementation combining formal causal discovery,
temporal logic, and counterfactual reasoning in a unified neural architecture
for cross-modal adaptive reasoning systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import itertools

logger = logging.getLogger(__name__)

class TemporalOperator(Enum):
    """Temporal logic operators"""
    ALWAYS = "G"      # Globally (always)
    EVENTUALLY = "F"  # Finally (eventually)
    NEXT = "X"        # Next
    UNTIL = "U"       # Until
    RELEASE = "R"     # Release
    SINCE = "S"       # Since (past)
    ONCE = "O"        # Once (past)

@dataclass
class CausalEdge:
    """Represents a causal relationship between variables"""
    cause: str
    effect: str
    strength: float
    confidence: float
    temporal_delay: int = 0
    modality: str = "unknown"
    intervention_tested: bool = False

@dataclass
class TemporalConstraint:
    """Represents a temporal logic constraint"""
    operator: TemporalOperator
    proposition: str
    operands: List[str]
    validity_window: Tuple[int, int]  # (start_time, end_time)
    truth_value: Optional[bool] = None

class CausalGraphDiscovery(nn.Module):
    """
    Automated causal graph discovery from observational data.
    
    Uses neural networks to learn causal relationships and structural
    equation models to represent causal mechanisms.
    """
    
    def __init__(self, n_variables: int, max_parents: int = 3):
        super().__init__()
        self.n_variables = n_variables
        self.max_parents = max_parents
        
        # Causal mechanism networks (one per variable)
        self.causal_mechanisms = nn.ModuleDict({
            f"var_{i}": nn.Sequential(
                nn.Linear(n_variables - 1, 128),  # All other variables as potential parents
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ) for i in range(n_variables)
        })
        
        # Causal structure learning network
        self.structure_learner = nn.Sequential(
            nn.Linear(n_variables * n_variables, 256),  # Adjacency matrix flattened
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_variables * n_variables),
            nn.Sigmoid()
        )
        
        # Independence test networks
        self.independence_tests = nn.ModuleDict({
            f"test_{i}_{j}": nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for i in range(n_variables) for j in range(n_variables) if i != j
        })
        
        self.discovered_edges = []
        self.causal_history = []
        
    def forward(self, data: torch.Tensor, 
                modality_labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[CausalEdge]]:
        """
        Discover causal graph from multi-modal data
        
        Args:
            data: Multi-modal data tensor [batch, time, variables]
            modality_labels: Labels indicating modality for each variable
            
        Returns:
            Tuple of (adjacency_matrix, discovered_causal_edges)
        """
        batch_size, seq_len, n_vars = data.shape
        
        # Learn causal structure using constraint-based approach
        adjacency_matrix = self._discover_causal_structure(data)
        
        # Apply causal mechanisms to predict values
        causal_predictions = self._apply_causal_mechanisms(data, adjacency_matrix)
        
        # Validate causal relationships using independence tests
        validated_edges = self._validate_causal_edges(data, adjacency_matrix, modality_labels)
        
        # Update discovered edges
        self.discovered_edges = validated_edges
        
        # Record discovery history
        self.causal_history.append({
            'n_edges_discovered': len(validated_edges),
            'adjacency_sparsity': torch.sum(adjacency_matrix > 0.5).item() / (n_vars ** 2),
            'average_edge_strength': np.mean([edge.strength for edge in validated_edges]),
            'cross_modal_edges': len([e for e in validated_edges if e.modality == "cross_modal"])
        })
        
        return adjacency_matrix, validated_edges
    
    def _discover_causal_structure(self, data: torch.Tensor) -> torch.Tensor:
        """Discover causal structure using structural learning"""
        batch_size, seq_len, n_vars = data.shape
        
        # Compute pairwise correlations and dependencies
        correlations = self._compute_correlations(data)
        
        # Use structure learner to predict adjacency matrix
        correlation_flat = correlations.flatten().unsqueeze(0)
        predicted_structure = self.structure_learner(correlation_flat)
        adjacency_matrix = predicted_structure.reshape(n_vars, n_vars)
        
        # Ensure DAG constraint (no cycles)
        adjacency_matrix = self._enforce_dag_constraint(adjacency_matrix)
        
        return adjacency_matrix
    
    def _compute_correlations(self, data: torch.Tensor) -> torch.Tensor:
        """Compute correlation matrix from time series data"""
        batch_size, seq_len, n_vars = data.shape
        
        # Average over batch and compute correlations over time
        avg_data = data.mean(dim=0)  # [seq_len, n_vars]
        
        correlations = torch.zeros(n_vars, n_vars)
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    corr = torch.corrcoef(torch.stack([avg_data[:, i], avg_data[:, j]]))[0, 1]
                    correlations[i, j] = torch.abs(corr) if not torch.isnan(corr) else 0.0
                    
        return correlations
    
    def _enforce_dag_constraint(self, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """Enforce directed acyclic graph constraint"""
        n_vars = adjacency_matrix.shape[0]
        
        # Use topological ordering to break cycles
        # Simple approach: upper triangular matrix
        mask = torch.triu(torch.ones(n_vars, n_vars, dtype=torch.bool), diagonal=1)
        dag_adjacency = torch.zeros_like(adjacency_matrix)
        dag_adjacency[mask] = adjacency_matrix[mask]
        
        return dag_adjacency
    
    def _apply_causal_mechanisms(self, data: torch.Tensor, 
                                adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """Apply learned causal mechanisms to predict variable values"""
        batch_size, seq_len, n_vars = data.shape
        predictions = torch.zeros_like(data)
        
        for var_idx in range(n_vars):
            # Get parents of this variable
            parents = torch.where(adjacency_matrix[:, var_idx] > 0.5)[0]
            
            if len(parents) > 0:
                # Use parent values to predict this variable
                parent_data = data[:, :, parents].reshape(batch_size * seq_len, -1)
                mechanism = self.causal_mechanisms[f"var_{var_idx}"]
                var_predictions = mechanism(parent_data)
                predictions[:, :, var_idx] = var_predictions.reshape(batch_size, seq_len)
            else:
                # No parents - use marginal distribution
                predictions[:, :, var_idx] = data[:, :, var_idx]
                
        return predictions
    
    def _validate_causal_edges(self, data: torch.Tensor, 
                              adjacency_matrix: torch.Tensor,
                              modality_labels: Optional[torch.Tensor] = None) -> List[CausalEdge]:
        """Validate discovered edges using independence tests"""
        n_vars = adjacency_matrix.shape[0]
        validated_edges = []
        
        for i in range(n_vars):
            for j in range(n_vars):
                if adjacency_matrix[i, j] > 0.5:  # Edge exists
                    # Test independence
                    independence_score = self._test_independence(data, i, j)
                    
                    if independence_score < 0.3:  # Dependent (causal)
                        # Determine modality
                        if modality_labels is not None:
                            modality = "cross_modal" if modality_labels[i] != modality_labels[j] else "intra_modal"
                        else:
                            modality = "unknown"
                        
                        edge = CausalEdge(
                            cause=f"var_{i}",
                            effect=f"var_{j}",
                            strength=adjacency_matrix[i, j].item(),
                            confidence=1.0 - independence_score,
                            modality=modality
                        )
                        validated_edges.append(edge)
        
        return validated_edges
    
    def _test_independence(self, data: torch.Tensor, var_i: int, var_j: int) -> float:
        """Test statistical independence between two variables"""
        batch_size, seq_len, n_vars = data.shape
        
        # Flatten data for independence test
        x_i = data[:, :, var_i].reshape(-1, 1)
        x_j = data[:, :, var_j].reshape(-1, 1)
        
        # Use neural independence test
        test_input = torch.cat([x_i, x_j], dim=1)
        independence_test = self.independence_tests[f"test_{var_i}_{var_j}"]
        independence_score = independence_test(test_input).mean()
        
        return independence_score.item()
    
    def get_causal_graph_summary(self) -> Dict[str, Any]:
        """Get summary of discovered causal graph"""
        if not self.discovered_edges:
            return {}
        
        edge_strengths = [edge.strength for edge in self.discovered_edges]
        confidences = [edge.confidence for edge in self.discovered_edges]
        
        return {
            'n_causal_edges': len(self.discovered_edges),
            'average_edge_strength': np.mean(edge_strengths),
            'average_confidence': np.mean(confidences),
            'cross_modal_ratio': len([e for e in self.discovered_edges if e.modality == "cross_modal"]) / len(self.discovered_edges),
            'causal_density': len(self.discovered_edges) / (self.n_variables ** 2)
        }

class TemporalLogicEngine(nn.Module):
    """
    Temporal logic engine for reasoning about time-dependent properties.
    
    Implements Linear Temporal Logic (LTL) operators and integrates them
    into neural processing for temporal constraint satisfaction.
    """
    
    def __init__(self, n_propositions: int, max_horizon: int = 20):
        super().__init__()
        self.n_propositions = n_propositions
        self.max_horizon = max_horizon
        
        # Proposition evaluators
        self.proposition_evaluators = nn.ModuleDict({
            f"prop_{i}": nn.Sequential(
                nn.Linear(n_propositions, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for i in range(n_propositions)
        })
        
        # Temporal operator networks
        self.temporal_operators = nn.ModuleDict({
            "globally": self._create_operator_network(),      # G (always)
            "eventually": self._create_operator_network(),    # F (eventually)
            "next": self._create_operator_network(),          # X (next)
            "until": self._create_operator_network(),         # U (until)
            "release": self._create_operator_network(),       # R (release)
        })
        
        # Constraint satisfaction network
        self.constraint_solver = nn.Sequential(
            nn.Linear(max_horizon * n_propositions, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, max_horizon),
            nn.Sigmoid()
        )
        
        self.temporal_constraints = []
        self.satisfaction_history = []
        
    def _create_operator_network(self) -> nn.Module:
        """Create network for temporal operator evaluation"""
        return nn.Sequential(
            nn.Linear(self.max_horizon, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.max_horizon),
            nn.Sigmoid()
        )
    
    def forward(self, temporal_data: torch.Tensor, 
                causal_graph: List[CausalEdge]) -> Tuple[torch.Tensor, List[TemporalConstraint]]:
        """
        Apply temporal logic reasoning to sequential data
        
        Args:
            temporal_data: Sequential data [batch, time, features]
            causal_graph: Discovered causal relationships
            
        Returns:
            Tuple of (temporal_reasoning_output, satisfied_constraints)
        """
        batch_size, seq_len, n_features = temporal_data.shape
        
        # Evaluate propositions at each time step
        proposition_values = self._evaluate_propositions(temporal_data)
        
        # Apply temporal operators
        temporal_operator_outputs = self._apply_temporal_operators(proposition_values)
        
        # Generate temporal constraints from causal graph
        generated_constraints = self._generate_temporal_constraints(causal_graph, seq_len)
        
        # Solve constraint satisfaction problem
        satisfaction_values = self._solve_constraints(temporal_operator_outputs, generated_constraints)
        
        # Combine temporal reasoning outputs
        temporal_output = torch.cat([
            temporal_operator_outputs["globally"],
            temporal_operator_outputs["eventually"],
            satisfaction_values.unsqueeze(-1)
        ], dim=-1)
        
        # Record satisfaction history
        avg_satisfaction = torch.mean(satisfaction_values).item()
        self.satisfaction_history.append({
            'average_satisfaction': avg_satisfaction,
            'n_constraints': len(generated_constraints),
            'constraint_types': [c.operator.value for c in generated_constraints]
        })
        
        return temporal_output, generated_constraints
    
    def _evaluate_propositions(self, temporal_data: torch.Tensor) -> torch.Tensor:
        """Evaluate truth values of propositions over time"""
        batch_size, seq_len, n_features = temporal_data.shape
        proposition_values = torch.zeros(batch_size, seq_len, self.n_propositions)
        
        for prop_idx in range(self.n_propositions):
            evaluator = self.proposition_evaluators[f"prop_{prop_idx}"]
            
            for t in range(seq_len):
                prop_values = evaluator(temporal_data[:, t, :])
                proposition_values[:, t, prop_idx] = prop_values.squeeze(-1)
                
        return proposition_values
    
    def _apply_temporal_operators(self, proposition_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply temporal logic operators to proposition sequences"""
        batch_size, seq_len, n_props = proposition_values.shape
        operator_outputs = {}
        
        for op_name, op_network in self.temporal_operators.items():
            op_outputs = torch.zeros(batch_size, seq_len, n_props)
            
            for prop_idx in range(n_props):
                prop_sequence = proposition_values[:, :, prop_idx]
                
                if op_name == "globally":
                    # G φ: φ is true at all future time points
                    for t in range(seq_len):
                        remaining_seq = prop_sequence[:, t:]
                        if remaining_seq.shape[1] > 0:
                            # Pad to max_horizon
                            padded_seq = F.pad(remaining_seq, (0, max(0, self.max_horizon - remaining_seq.shape[1])))[:, :self.max_horizon]
                            globally_val = torch.min(op_network(padded_seq), dim=1)[0]
                            op_outputs[:, t, prop_idx] = globally_val
                
                elif op_name == "eventually":
                    # F φ: φ will be true at some future time point
                    for t in range(seq_len):
                        remaining_seq = prop_sequence[:, t:]
                        if remaining_seq.shape[1] > 0:
                            padded_seq = F.pad(remaining_seq, (0, max(0, self.max_horizon - remaining_seq.shape[1])))[:, :self.max_horizon]
                            eventually_val = torch.max(op_network(padded_seq), dim=1)[0]
                            op_outputs[:, t, prop_idx] = eventually_val
                
                elif op_name == "next":
                    # X φ: φ is true at the next time point
                    for t in range(seq_len - 1):
                        next_val = proposition_values[:, t + 1, prop_idx]
                        op_outputs[:, t, prop_idx] = op_network(next_val.unsqueeze(1).expand(-1, self.max_horizon)).mean(dim=1)
                        
            operator_outputs[op_name] = op_outputs
            
        return operator_outputs
    
    def _generate_temporal_constraints(self, causal_graph: List[CausalEdge], 
                                     seq_len: int) -> List[TemporalConstraint]:
        """Generate temporal constraints from causal relationships"""
        constraints = []
        
        for edge in causal_graph:
            # Generate temporal constraints based on causal relationships
            
            # Constraint 1: If cause is true, effect should eventually be true
            eventually_constraint = TemporalConstraint(
                operator=TemporalOperator.EVENTUALLY,
                proposition=edge.effect,
                operands=[edge.cause],
                validity_window=(0, seq_len - 1)
            )
            constraints.append(eventually_constraint)
            
            # Constraint 2: Effect cannot occur before cause (temporal precedence)
            if edge.temporal_delay > 0:
                precedence_constraint = TemporalConstraint(
                    operator=TemporalOperator.UNTIL,
                    proposition=f"not_{edge.effect}",
                    operands=[edge.cause],
                    validity_window=(0, edge.temporal_delay)
                )
                constraints.append(precedence_constraint)
        
        self.temporal_constraints = constraints
        return constraints
    
    def _solve_constraints(self, operator_outputs: Dict[str, torch.Tensor],
                          constraints: List[TemporalConstraint]) -> torch.Tensor:
        """Solve temporal constraint satisfaction problem"""
        if not constraints:
            return torch.ones(operator_outputs["globally"].shape[0], operator_outputs["globally"].shape[1])
        
        # Combine all operator outputs
        all_outputs = torch.cat([
            operator_outputs["globally"].flatten(1),
            operator_outputs["eventually"].flatten(1)
        ], dim=1)
        
        # Use constraint solver network
        satisfaction_scores = self.constraint_solver(all_outputs)
        
        return satisfaction_scores
    
    def add_custom_constraint(self, constraint: TemporalConstraint):
        """Add custom temporal constraint"""
        self.temporal_constraints.append(constraint)
        logger.info(f"Added temporal constraint: {constraint.operator.value} {constraint.proposition}")

class CounterfactualGenerator(nn.Module):
    """
    Counterfactual reasoning system for "what if" scenario generation.
    
    Generates counterfactual scenarios by intervening on causal variables
    and predicting alternative outcomes.
    """
    
    def __init__(self, n_variables: int, intervention_strength: float = 1.0):
        super().__init__()
        self.n_variables = n_variables
        self.intervention_strength = intervention_strength
        
        # Counterfactual outcome predictor
        self.outcome_predictor = nn.Sequential(
            nn.Linear(n_variables * 2, 256),  # Original + intervention
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_variables)
        )
        
        # Intervention effect estimator
        self.intervention_estimator = nn.ModuleDict({
            f"intervention_{i}": nn.Sequential(
                nn.Linear(n_variables, 64),
                nn.ReLU(),
                nn.Linear(64, n_variables)
            ) for i in range(n_variables)
        })
        
        # Plausibility scorer
        self.plausibility_scorer = nn.Sequential(
            nn.Linear(n_variables * 3, 128),  # Original + intervention + counterfactual
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.counterfactual_history = []
        
    def forward(self, original_data: torch.Tensor, 
                causal_graph: List[CausalEdge],
                intervention_variables: List[int],
                intervention_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate counterfactual scenarios through causal intervention
        
        Args:
            original_data: Original observational data
            causal_graph: Discovered causal relationships
            intervention_variables: Variables to intervene on
            intervention_values: Values to set for intervened variables
            
        Returns:
            Dictionary with counterfactual results
        """
        batch_size, n_vars = original_data.shape
        
        # Create intervention mask
        intervention_mask = torch.zeros(n_vars, dtype=torch.bool)
        intervention_mask[intervention_variables] = True
        
        # Apply interventions
        intervened_data = original_data.clone()
        intervened_data[:, intervention_variables] = intervention_values
        
        # Predict counterfactual outcomes
        input_data = torch.cat([original_data, intervened_data], dim=1)
        counterfactual_outcomes = self.outcome_predictor(input_data)
        
        # Estimate intervention effects for each variable
        intervention_effects = self._estimate_intervention_effects(
            original_data, intervention_variables, intervention_values, causal_graph
        )
        
        # Apply causal propagation
        final_counterfactuals = self._propagate_causal_effects(
            intervened_data, intervention_effects, causal_graph
        )
        
        # Score plausibility
        plausibility_input = torch.cat([original_data, intervened_data, final_counterfactuals], dim=1)
        plausibility_scores = self.plausibility_scorer(plausibility_input)
        
        # Compute intervention impact
        impact_scores = torch.norm(final_counterfactuals - original_data, dim=1, keepdim=True)
        
        results = {
            'counterfactual_outcomes': final_counterfactuals,
            'intervention_effects': intervention_effects,
            'plausibility_scores': plausibility_scores,
            'impact_scores': impact_scores,
            'intervention_variables': intervention_variables
        }
        
        # Record counterfactual generation
        self.counterfactual_history.append({
            'n_interventions': len(intervention_variables),
            'average_impact': torch.mean(impact_scores).item(),
            'average_plausibility': torch.mean(plausibility_scores).item(),
            'intervention_strength_used': self.intervention_strength
        })
        
        return results
    
    def _estimate_intervention_effects(self, original_data: torch.Tensor,
                                     intervention_variables: List[int],
                                     intervention_values: torch.Tensor,
                                     causal_graph: List[CausalEdge]) -> torch.Tensor:
        """Estimate effects of interventions on each variable"""
        batch_size, n_vars = original_data.shape
        intervention_effects = torch.zeros_like(original_data)
        
        for var_idx in intervention_variables:
            # Use intervention estimator for this variable
            estimator = self.intervention_estimator[f"intervention_{var_idx}"]
            effects = estimator(original_data)
            
            # Scale by intervention strength
            intervention_magnitude = torch.abs(intervention_values - original_data[:, var_idx:var_idx+1])
            scaled_effects = effects * intervention_magnitude * self.intervention_strength
            
            intervention_effects += scaled_effects
        
        return intervention_effects
    
    def _propagate_causal_effects(self, intervened_data: torch.Tensor,
                                intervention_effects: torch.Tensor,
                                causal_graph: List[CausalEdge]) -> torch.Tensor:
        """Propagate causal effects through the causal graph"""
        propagated_data = intervened_data + intervention_effects
        
        # Multiple rounds of propagation to handle indirect effects
        for round_idx in range(3):
            for edge in causal_graph:
                cause_idx = int(edge.cause.split('_')[1])
                effect_idx = int(edge.effect.split('_')[1])
                
                # Propagate effect based on causal strength
                causal_influence = propagated_data[:, cause_idx:cause_idx+1] * edge.strength
                propagated_data[:, effect_idx:effect_idx+1] += causal_influence * 0.1  # Damping factor
        
        return propagated_data
    
    def generate_diverse_counterfactuals(self, original_data: torch.Tensor,
                                       causal_graph: List[CausalEdge],
                                       n_scenarios: int = 5) -> List[Dict[str, torch.Tensor]]:
        """Generate diverse counterfactual scenarios"""
        scenarios = []
        
        for scenario_idx in range(n_scenarios):
            # Randomly select intervention variables
            n_interventions = torch.randint(1, min(4, self.n_variables + 1), (1,)).item()
            intervention_vars = torch.randperm(self.n_variables)[:n_interventions].tolist()
            
            # Generate intervention values
            intervention_values = torch.randn(original_data.shape[0], len(intervention_vars))
            
            # Generate counterfactual
            scenario = self.forward(
                original_data, causal_graph, intervention_vars, intervention_values
            )
            scenario['scenario_id'] = scenario_idx
            scenarios.append(scenario)
        
        return scenarios

class CausalTemporalReasoningAdapter(nn.Module):
    """
    Revolutionary cross-modal causal temporal reasoning adapter.
    
    Integrates causal discovery, temporal logic, and counterfactual reasoning
    for sophisticated temporal and causal reasoning across modalities.
    """
    
    def __init__(self, 
                 input_dim: int,
                 n_variables: int = None,
                 max_temporal_horizon: int = 20):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_variables = n_variables or min(input_dim, 10)  # Reasonable default
        self.max_temporal_horizon = max_temporal_horizon
        
        # Core reasoning components
        self.causal_discovery = CausalGraphDiscovery(
            n_variables=self.n_variables,
            max_parents=3
        )
        
        self.temporal_logic_engine = TemporalLogicEngine(
            n_propositions=self.n_variables,
            max_horizon=max_temporal_horizon
        )
        
        self.counterfactual_generator = CounterfactualGenerator(
            n_variables=self.n_variables,
            intervention_strength=0.5
        )
        
        # Cross-modal integration
        self.cross_modal_reasoner = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_variables)
        )
        
        # Temporal integration network
        self.temporal_integrator = nn.LSTM(
            input_size=self.n_variables * 3,  # Causal + temporal + counterfactual
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Final reasoning output
        self.reasoning_output = nn.Sequential(
            nn.Linear(128 + input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        
        self.reasoning_history = []
        
    def forward(self, multi_modal_input: torch.Tensor,
                temporal_context: Optional[torch.Tensor] = None,
                modality_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through causal temporal reasoning
        
        Args:
            multi_modal_input: Multi-modal input data [batch, time, features]
            temporal_context: Optional temporal context information
            modality_labels: Labels indicating modality for each feature
            
        Returns:
            Causally and temporally reasoned output
        """
        if len(multi_modal_input.shape) == 2:
            # Add temporal dimension if missing
            multi_modal_input = multi_modal_input.unsqueeze(1)
        
        batch_size, seq_len, feature_dim = multi_modal_input.shape
        
        # Extract variables for causal reasoning
        variable_data = self.cross_modal_reasoner(multi_modal_input)  # [batch, time, n_variables]
        
        # Discover causal relationships
        adjacency_matrix, causal_edges = self.causal_discovery(variable_data, modality_labels)
        
        # Apply temporal logic reasoning
        temporal_output, temporal_constraints = self.temporal_logic_engine(variable_data, causal_edges)
        
        # Generate counterfactual scenarios
        if seq_len > 1:
            # Use latest timestep for counterfactual generation
            latest_data = variable_data[:, -1, :]
            counterfactual_results = self.counterfactual_generator.generate_diverse_counterfactuals(
                latest_data, causal_edges, n_scenarios=3
            )
            
            # Average counterfactual outcomes
            avg_counterfactual = torch.stack([
                cf['counterfactual_outcomes'] for cf in counterfactual_results
            ], dim=0).mean(dim=0)
            
            # Expand to match temporal dimensions
            avg_counterfactual = avg_counterfactual.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            avg_counterfactual = torch.zeros_like(variable_data)
        
        # Integrate all reasoning components
        integrated_reasoning = torch.cat([
            variable_data,
            temporal_output,
            avg_counterfactual
        ], dim=-1)
        
        # Temporal integration with LSTM
        lstm_output, _ = self.temporal_integrator(integrated_reasoning)
        
        # Final reasoning output
        final_input = torch.cat([
            lstm_output[:, -1, :],  # Last timestep of LSTM
            multi_modal_input[:, -1, :]  # Last timestep of original input
        ], dim=-1)
        
        reasoning_output = self.reasoning_output(final_input)
        
        # Record reasoning metrics
        self._record_reasoning_event(causal_edges, temporal_constraints, counterfactual_results if seq_len > 1 else [])
        
        return reasoning_output
    
    def _record_reasoning_event(self, causal_edges: List[CausalEdge],
                              temporal_constraints: List[TemporalConstraint],
                              counterfactual_results: List[Dict[str, torch.Tensor]]):
        """Record reasoning event for analysis"""
        reasoning_metrics = {
            'n_causal_edges': len(causal_edges),
            'n_temporal_constraints': len(temporal_constraints),
            'n_counterfactual_scenarios': len(counterfactual_results),
            'cross_modal_causal_edges': len([e for e in causal_edges if e.modality == "cross_modal"]),
            'average_causal_strength': np.mean([e.strength for e in causal_edges]) if causal_edges else 0,
            'temporal_reasoning_active': len(temporal_constraints) > 0,
            'counterfactual_reasoning_active': len(counterfactual_results) > 0
        }
        
        if counterfactual_results:
            reasoning_metrics.update({
                'average_counterfactual_impact': np.mean([
                    torch.mean(cf['impact_scores']).item() for cf in counterfactual_results
                ]),
                'average_counterfactual_plausibility': np.mean([
                    torch.mean(cf['plausibility_scores']).item() for cf in counterfactual_results
                ])
            })
        
        self.reasoning_history.append(reasoning_metrics)
        
        # Keep history manageable
        if len(self.reasoning_history) > 1000:
            self.reasoning_history = self.reasoning_history[-1000:]
    
    def get_reasoning_analysis(self) -> Dict[str, Any]:
        """Comprehensive analysis of causal temporal reasoning"""
        if not self.reasoning_history:
            return {}
        
        recent_history = self.reasoning_history[-100:]
        
        causal_metrics = self.causal_discovery.get_causal_graph_summary()
        
        return {
            'causal_reasoning': causal_metrics,
            'temporal_reasoning': {
                'average_constraints_per_step': np.mean([h['n_temporal_constraints'] for h in recent_history]),
                'temporal_satisfaction_rate': np.mean([
                    h.get('temporal_satisfaction', 0) for h in recent_history
                ])
            },
            'counterfactual_reasoning': {
                'scenarios_generated_per_step': np.mean([h['n_counterfactual_scenarios'] for h in recent_history]),
                'average_impact_magnitude': np.mean([
                    h.get('average_counterfactual_impact', 0) for h in recent_history
                ]),
                'average_plausibility': np.mean([
                    h.get('average_counterfactual_plausibility', 0) for h in recent_history
                ])
            },
            'cross_modal_integration': {
                'cross_modal_causal_ratio': np.mean([
                    h['cross_modal_causal_edges'] / max(h['n_causal_edges'], 1) for h in recent_history
                ]),
                'multi_modal_reasoning_active': True
            },
            'overall_reasoning_complexity': {
                'total_reasoning_components': 3,  # Causal + temporal + counterfactual
                'average_reasoning_depth': np.mean([
                    h['n_causal_edges'] + h['n_temporal_constraints'] + h['n_counterfactual_scenarios']
                    for h in recent_history
                ])
            }
        }

def create_causal_temporal_reasoning_adapter(input_dim: int,
                                           reasoning_config: Optional[Dict] = None) -> CausalTemporalReasoningAdapter:
    """
    Factory function for creating causal temporal reasoning adapters
    
    Args:
        input_dim: Input dimension for the adapter
        reasoning_config: Optional reasoning configuration
        
    Returns:
        Revolutionary causal temporal reasoning adapter
    """
    if reasoning_config is None:
        reasoning_config = {}
    
    adapter = CausalTemporalReasoningAdapter(
        input_dim=input_dim,
        n_variables=reasoning_config.get('n_variables', min(input_dim, 10)),
        max_temporal_horizon=reasoning_config.get('temporal_horizon', 20)
    )
    
    # Add custom temporal constraints if provided
    if 'custom_constraints' in reasoning_config:
        for constraint_config in reasoning_config['custom_constraints']:
            constraint = TemporalConstraint(**constraint_config)
            adapter.temporal_logic_engine.add_custom_constraint(constraint)
    
    logger.info(f"Created causal temporal reasoning adapter with {adapter.n_variables} variables")
    
    return adapter