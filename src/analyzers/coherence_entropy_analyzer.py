from typing import List, Dict, Any
import numpy as np
from qiskit.quantum_info import Statevector, state_fidelity
import matplotlib.pyplot as plt

class EnhancedCoherenceEntropyAnalyzer:
    """Advanced analyzer for coherence-entropy relationships in emotional transitions."""
    
    def __init__(self):
        self.coherence_data = []
        self.entropy_data = []
        self.balance_data = []
        self.critical_points = []
        self.transition_patterns = []
        self.stability_metrics = []
    
    def analyze_transition(self, state_history: List[Statevector]) -> Dict[str, Any]:
        """
        Performs enhanced analysis of coherence-entropy balance during emotional transition.
        
        This analysis uses quantum-inspired concepts to explore how emotional states 
        transition, focusing on the balance between coherence (representing emotional 
        clarity or consistency) and entropy (representing emotional complexity). 

        Args:
            state_history: List of quantum-inspired states during transition
            
        Returns:
            Dictionary containing detailed analysis results
        """
        n_steps = len(state_history)
        local_window = 3  # Window size for local analysis
        
        for step, state in enumerate(state_history):
            # Calculate primary metrics
            coherence = self._calculate_coherence(state)
            entropy = self._calculate_entropy(state)
            balance = self._calculate_balance(coherence, entropy)
            
            # Track data
            self.coherence_data.append(coherence)
            self.entropy_data.append(entropy)
            self.balance_data.append(balance)
            
            # Analyze local patterns
            if step >= local_window - 1:
                window_data = {
                    'coherence': self.coherence_data[step-local_window+1:step+1],
                    'entropy': self.entropy_data[step-local_window+1:step+1],
                    'balance': self.balance_data[step-local_window+1:step+1]
                }
                
                # Detect critical points using local analysis
                if self._is_critical_point(window_data):
                    critical_point = self._analyze_critical_point(
                        step, state, state_history[max(0, step-1)], window_data
                    )
                    self.critical_points.append(critical_point)
            
            # Track transition patterns and stability
            if step > 0:
                pattern = self._analyze_transition_pattern(
                    step, state, state_history[step-1]
                )
                self.transition_patterns.append(pattern)
                self.stability_metrics.append(
                    self._calculate_stability(step, state_history[:step+1])
                )
        
        return {
            'coherence_data': self.coherence_data,
            'entropy_data': self.entropy_data,
            'balance_data': self.balance_data,
            'critical_points': self.critical_points,
            'transition_patterns': self.transition_patterns,
            'stability_metrics': self.stability_metrics,
            'summary': self._generate_summary()
        }
    
    def _calculate_coherence(self, state: Statevector) -> float:
        """
        Calculates enhanced L1-norm coherence with regularization as a metaphor for emotional clarity.

        The L1-norm coherence here represents how 'pure' or 'clear' an emotional state is. 
        By normalizing to the dimension of the state, we ensure comparability across different 
        state sizes. This is inspired by the concept of emotional clarity [Salovey et al., 1995], 
        where higher clarity corresponds to a more coherent emotional state.

        Args:
            state: Quantum-inspired state

        Returns:
            Float representing normalized emotional coherence (clarity)
        """
        density_matrix = state.to_operator().data
        off_diagonal = np.abs(density_matrix - np.diag(np.diag(density_matrix)))
        return np.sum(off_diagonal) / (state.dim ** 2)  # Normalized coherence
    
    def _calculate_entropy(self, state: Statevector) -> float:
        """
        Calculates regularized von Neumann entropy as a metaphor for emotional complexity.

        This metric captures the mixture or complexity of the emotional state, similar to 
        how entropy in quantum mechanics describes the disorder. In psychology, this could 
        be linked to the concept of emotional complexity [Kashdan et al., 2015], where 
        higher entropy might indicate more nuanced or mixed emotional experiences. 
        Regularization avoids issues with zero eigenvalues.

        Args:
            state: Quantum-inspired state

        Returns:
            Float representing normalized emotional entropy (complexity)
        """
        density_matrix = state.to_operator().data
        eigenvals = np.real(np.linalg.eigvals(density_matrix))
        eigenvals = np.clip(eigenvals, 1e-10, 1.0)
        max_entropy = np.log2(state.dim)
        return -np.sum(eigenvals * np.log2(eigenvals)) / max_entropy
    
    def _calculate_balance(self, coherence: float, entropy: float) -> float:
        """
        Calculates enhanced balance measure with normalization as a metaphor for emotional balance.

        This balance measure represents the interplay between emotional clarity (coherence) 
        and complexity (entropy). A higher balance indicates an emotional state where clarity 
        is maintained despite complexity. This concept draws from the idea of emotional 
        regulation where balance might be seen as how well one manages emotional complexity 
        [Gross, 1998].

        Args:
            coherence: Emotional coherence (clarity)
            entropy: Emotional entropy (complexity)

        Returns:
            Float representing the emotional balance
        """
        return (coherence * (1 - entropy) + 1e-10) / (1 + entropy)
    
    def _is_critical_point(self, window_data: Dict[str, List[float]]) -> bool:
        """
        Enhanced critical point detection using multiple criteria.

        Critical points are moments where significant changes in the emotional state occur, 
        possibly indicating transitions or pivotal moments in emotional experience. Here, 
        we look for changes in the rate of change of balance, coherence, and entropy, 
        which might signal shifts in emotional regulation or state clarity [Tugade & Fredrickson, 2004].

        Args:
            window_data: Dictionary containing recent window of coherence, entropy, and balance data

        Returns:
            Boolean indicating if current step is a critical point
        """
        balance_derivative = np.gradient(window_data['balance'])
        coherence_derivative = np.gradient(window_data['coherence'])
        entropy_derivative = np.gradient(window_data['entropy'])
        
        balance_std = np.std(window_data['balance'])
        coherence_std = np.std(window_data['coherence'])
        entropy_std = np.std(window_data['entropy'])
        
        balance_threshold = 2 * balance_std
        metric_threshold = np.mean([coherence_std, entropy_std])
        
        return (np.abs(balance_derivative[-1]) > balance_threshold or
                np.abs(coherence_derivative[-1]) > metric_threshold or
                np.abs(entropy_derivative[-1]) > metric_threshold)
    
    def _analyze_critical_point(self, step: int, current_state: Statevector,
                              previous_state: Statevector, 
                              window_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Analyzes characteristics of a critical point.

        At critical points, we look at how much the emotional state has shifted, which could 
        reflect moments of significant emotional change or regulation. This analysis helps 
        understand the dynamics of emotional transitions, inspired by studies on emotional 
        transitions and coping mechanisms [Folkman & Moskowitz, 2000].

        Args:
            step: Current step in transition
            current_state: Current quantum-inspired emotional state
            previous_state: Previous quantum-inspired emotional state
            window_data: Local window data for analysis

        Returns:
            Dictionary containing analysis of the critical point
        """
        return {
            'step': step,
            'coherence_change': self.coherence_data[step] - self.coherence_data[step-1],
            'entropy_change': self.entropy_data[step] - self.entropy_data[step-1],
            'balance_change': self.balance_data[step] - self.balance_data[step-1],
            'local_stability': self._calculate_stability(step, [current_state, previous_state])
        }
    
    def _analyze_transition_pattern(self, step: int, current_state: Statevector,
                                  previous_state: Statevector) -> Dict[str, Any]:
        """
        Analyzes transition patterns between states.

        This function looks at how balance changes from one emotional state to the next, 
        providing insight into the flow or stability of emotional transitions. Stability 
        here can be related to emotional inertia or resilience [Kuppens et al., 2010].

        Args:
            step: Current step in transition
            current_state: Current quantum-inspired emotional state
            previous_state: Previous quantum-inspired emotional state

        Returns:
            Dictionary containing transition pattern data
        """
        return {
            'step': step,
            'balance_change': self.balance_data[step] - self.balance_data[step-1],
            'stability': self._calculate_stability(step, [current_state, previous_state])
        }
    
    def _calculate_stability(self, step: int, state_history: List[Statevector]) -> float:
        """
        Calculates stability metric for current state.

        Stability in this context metaphorically represents how similar the current emotional 
        state is to the previous one, drawing parallels with emotional stability where 
        high fidelity might suggest emotional resilience or consistency [Ong et al., 2006].

        Args:
            step: Current step in transition
            state_history: History of quantum-inspired states up to this step

        Returns:
            Float representing the stability of the current emotional state
        """
        if step == 0 or len(state_history) < 2:
            return 1.0
        return state_fidelity(state_history[-1], state_history[-2])
    
    def _generate_summary(self) -> Dict[str, float]:
        """Generates summary statistics of analysis."""
        return {
            'total_critical_points': len(self.critical_points),
            'max_coherence': max(self.coherence_data),
            'min_entropy': min(self.entropy_data),
            'max_balance': max(self.balance_data),
            'min_balance': min(self.balance_data),
            'avg_stability': np.mean(self.stability_metrics)
        }
    
    def visualize_analysis(self, analysis_results: Dict[str, Any]) -> None:
        """Visualizes the analysis results."""
        steps = range(len(analysis_results['coherence_data']))
        
        fig = plt.figure(figsize=(15, 18))
        
        # Coherence-Entropy Evolution
        ax1 = fig.add_subplot(321)
        ax1.plot(steps, analysis_results['coherence_data'], 'b-', label='Coherence (Clarity)')
        ax1.plot(steps, analysis_results['entropy_data'], 'r--', label='Entropy (Complexity)')
        for cp in analysis_results['critical_points']:
            ax1.axvline(x=cp['step'], color='g', linestyle='--', alpha=0.5, label='Critical Point' if cp['step'] == analysis_results['critical_points'][0]['step'] else "")
        ax1.set_title('Coherence-Entropy Evolution in Emotional Transition')
        ax1.set_xlabel('Transition Step')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True)
        
        # Balance Evolution
        ax2 = fig.add_subplot(322)
        ax2.plot(steps, analysis_results['balance_data'], 'g-', linewidth=2)
        for cp in analysis_results['critical_points']:
            ax2.axvline(x=cp['step'], color='r', linestyle='--', alpha=0.5, label='Critical Point' if cp['step'] == analysis_results['critical_points'][0]['step'] else "")
        ax2.set_title('Emotional Balance Evolution')
        ax2.setSorry about that, something didn't go as planned. Please try again, and if you're still seeing this message, go ahead and restart the app.
