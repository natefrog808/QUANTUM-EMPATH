from typing import List, Dict, Any
import numpy as np
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

class CriticalBalanceAnalyzer:
    """Analyzes critical balance points in coherence-entropy relationships."""
    
    def __init__(self):
        self.balance_points = []
        self.stability_metrics = []
        self.transition_patterns = []
    
    def analyze_critical_balance_points(self, state_history: List[Statevector]) -> Dict[str, Any]:
        """
        Analyzes critical points where coherence-entropy balance shows significant changes.
        
        Args:
            state_history: List of quantum states during transition
            
        Returns:
            Dictionary containing critical balance point analysis
        """
        n_steps = len(state_history)
        balance_data = []
        critical_points = []
        
        # Initialize tracking metrics
        coherence_history = []
        entropy_history = []
        balance_history = []
        
        for step, state in enumerate(state_history):
            # Calculate quantum metrics
            coherence = self._calculate_coherence(state)
            entropy = self._calculate_entropy(state)
            balance = self._calculate_balance(coherence, entropy)
            
            # Store histories
            coherence_history.append(coherence)
            entropy_history.append(entropy)
            balance_history.append(balance)
            
            # Calculate derivatives if enough history
            if step > 0:
                coherence_rate = coherence - coherence_history[-2]
                entropy_rate = entropy - entropy_history[-2]
                balance_rate = balance - balance_history[-2]
                
                # Identify critical points
                if self._is_critical_point(coherence_rate, entropy_rate, balance_rate):
                    critical_points.append({
                        'step': step,
                        'coherence': coherence,
                        'entropy': entropy,
                        'balance': balance,
                        'coherence_rate': coherence_rate,
                        'entropy_rate': entropy_rate,
                        'balance_rate': balance_rate,
                        'stability': self._calculate_stability(balance_history[-3:])
                    })
            
            # Store balance data
            balance_data.append({
                'step': step,
                'coherence': coherence,
                'entropy': entropy,
                'balance': balance
            })
        
        return {
            'balance_data': balance_data,
            'critical_points': critical_points,
            'summary': self._generate_summary(balance_data, critical_points),
            'transition_patterns': self._analyze_transition_patterns(critical_points)
        }
    
    def _calculate_coherence(self, state: Statevector) -> float:
        """Calculates quantum coherence with improved accuracy."""
        density_matrix = state.to_operator().data
        diagonal = np.diag(np.diag(density_matrix))
        return np.sum(np.abs(density_matrix - diagonal))
    
    def _calculate_entropy(self, state: Statevector) -> float:
        """Calculates von Neumann entropy with stability enhancements."""
        density_matrix = state.to_operator().data
        eigenvals = np.real(np.linalg.eigvals(density_matrix))
        eigenvals = np.clip(eigenvals, 1e-10, 1.0)
        return -np.sum(eigenvals * np.log2(eigenvals))
    
    def _calculate_balance(self, coherence: float, entropy: float) -> float:
        """Calculates enhanced balance measure between coherence and entropy."""
        max_entropy = np.log2(4)  # Maximum possible entropy for 2-qubit system
        normalized_entropy = entropy / max_entropy
        return coherence * (1 - normalized_entropy)
    
    def _is_critical_point(self, coherence_rate: float, 
                          entropy_rate: float, 
                          balance_rate: float) -> bool:
        """Determines if a point is critical using multiple criteria."""
        # Define thresholds
        coherence_threshold = 0.1
        entropy_threshold = 0.1
        balance_threshold = 0.1
        
        # Check multiple conditions
        significant_coherence = abs(coherence_rate) > coherence_threshold
        significant_entropy = abs(entropy_rate) > entropy_threshold
        significant_balance = abs(balance_rate) > balance_threshold
        
        # Point is critical if multiple significant changes occur
        return (significant_coherence and significant_entropy) or \
               (significant_balance and (significant_coherence or significant_entropy))
    
    def _calculate_stability(self, balance_window: List[float]) -> float:
        """Calculates stability metric for balance points."""
        return np.std(balance_window) if len(balance_window) > 0 else 0
    
    def _analyze_transition_patterns(self, critical_points: List[Dict]) -> Dict[str, Any]:
        """Analyzes patterns in transitions between critical points."""
        if len(critical_points) < 2:
            return {}
        
        transitions = []
        for i in range(len(critical_points) - 1):
            current = critical_points[i]
            next_point = critical_points[i + 1]
            
            transitions.append({
                'start_step': current['step'],
                'end_step': next_point['step'],
                'coherence_change': next_point['coherence'] - current['coherence'],
                'entropy_change': next_point['entropy'] - current['entropy'],
                'balance_change': next_point['balance'] - current['balance'],
                'duration': next_point['step'] - current['step']
            })
        
        return {
            'transitions': transitions,
            'avg_duration': np.mean([t['duration'] for t in transitions]),
            'max_balance_change': max([abs(t['balance_change']) for t in transitions]),
            'pattern_stability': np.std([t['duration'] for t in transitions])
        }
    
    def _generate_summary(self, balance_data: List[Dict], 
                         critical_points: List[Dict]) -> Dict[str, Any]:
        """Generates summary statistics of critical points."""
        return {
            'n_critical_points': len(critical_points),
            'avg_balance': np.mean([d['balance'] for d in balance_data]),
            'max_balance': max([d['balance'] for d in balance_data]),
            'min_balance': min([d['balance'] for d in balance_data]),
            'avg_coherence': np.mean([d['coherence'] for d in balance_data]),
            'avg_entropy': np.mean([d['entropy'] for d in balance_data])
        }
    
    def visualize_critical_analysis(self, analysis_results: Dict[str, Any]) -> None:
        """Visualizes critical balance point analysis."""
        balance_data = analysis_results['balance_data']
        critical_points = analysis_results['critical_points']
        steps = range(len(balance_data))
        
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Coherence-Entropy Evolution with Critical Points
        ax1 = fig.add_subplot(321)
        coherence = [d['coherence'] for d in balance_data]
        entropy = [d['entropy'] for d in balance_data]
        ax1.plot(steps, coherence, 'b-', label='Coherence')
        ax1.plot(steps, entropy, 'r--', label='Entropy')
        # Mark critical points
        critical_steps = [cp['step'] for cp in critical_points]
        ax1.scatter(critical_steps, [coherence[s] for s in critical_steps], 
                   color='g', s=100, marker='o', label='Critical Points')
        ax1.set_title('Coherence-Entropy Evolution')
        ax1.set_xlabel('Transition Step')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Balance Evolution
        ax2 = fig.add_subplot(322)
        balance = [d['balance'] for d in balance_data]
        ax2.plot(steps, balance, 'g-', linewidth=2)
        # Mark critical points
        ax2.scatter(critical_steps, [balance[s] for s in critical_steps],
                   color='r', s=100, marker='o')
        ax2.set_title('Balance Evolution')
        ax2.set_xlabel('Transition Step')
        ax2.set_ylabel('Balance')
        ax2.grid(True)
        
        # 3. Critical Point Characteristics
        if critical_points:
            ax3 = fig.add_subplot(323)
            stability = [cp['stability'] for cp in critical_points]
            ax3.bar(range(len(critical_points)), stability, color='purple', alpha=0.7)
            ax3.set_title('Stability at Critical Points')
            ax3.set_xlabel('Critical Point Index')
            ax3.set_ylabel('Stability')
            
            # 4. Transition Pattern Analysis
            if 'transition_patterns' in analysis_results and analysis_results['transition_patterns']:
                ax4 = fig.add_subplot(324)
                transitions = analysis_results['transition_patterns']['transitions']
                durations = [t['duration'] for t in transitions]
                balance_changes = [t['balance_change'] for t in transitions]
                ax4.scatter(durations, balance_changes, color='orange', s=100)
                ax4.set_title('Transition Patterns')
                ax4.set_xlabel('Duration')
                ax4.set_ylabel('Balance Change')
                ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print analysis summary
        print("\nCritical Balance Point Analysis Summary:")
        summary = analysis_results['summary']
        print(f"Number of Critical Points: {summary['n_critical_points']}")
        print(f"Average Balance: {summary['avg_balance']:.4f}")
        print(f"Maximum Balance: {summary['max_balance']:.4f}")
        print(f"Average Coherence: {summary['avg_coherence']:.4f}")
        print(f"Average Entropy: {summary['avg_entropy']:.4f}")
        
        if 'transition_patterns' in analysis_results:
            patterns = analysis_results['transition_patterns']
            print(f"\nTransition Patterns:")
            print(f"Average Duration: {patterns['avg_duration']:.2f} steps")
            print(f"Maximum Balance Change: {patterns['max_balance_change']:.4f}")
            print(f"Pattern Stability: {patterns['pattern_stability']:.4f}")
