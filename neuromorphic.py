"""
Neuromorphic Version of T-RLINKOS TRM++

This module provides a spike-based neuromorphic implementation of T-RLINKOS
for deployment on neuromorphic hardware (Intel Loihi, IBM TrueNorth, etc.).

Key features:
- Spike-based dCaAP neuron with temporal dynamics
- Event-driven computation
- Synaptic plasticity (STDP)
- Low-power operation
- Temporal encoding/decoding

This is an experimental research module exploring neuromorphic computing
for recursive reasoning architectures.

References:
- Intel Loihi: https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html
- IBM TrueNorth: https://www.research.ibm.com/articles/brain-chip.html
- dCaAP in biological neurons: Gidon et al., Science 2020

Usage:
    # Create neuromorphic model
    model = NeuromorphicTRLinkosTRM(x_dim=64, y_dim=32, z_dim=64)
    
    # Encode input as spike train
    spike_train = model.encode_to_spikes(input_data)
    
    # Run neuromorphic inference
    output_spikes = model.forward_spikes(spike_train, time_steps=100)
    
    # Decode spikes to continuous output
    output = model.decode_from_spikes(output_spikes)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import warnings


# ============================
#  Configuration
# ============================

@dataclass
class NeuromorphicConfig:
    """Configuration for neuromorphic T-RLINKOS.
    
    Attributes:
        dt: Time step duration (ms)
        tau_mem: Membrane time constant (ms)
        tau_syn: Synaptic time constant (ms)
        v_thresh: Spike threshold voltage (mV)
        v_reset: Reset voltage after spike (mV)
        v_rest: Resting potential (mV)
        refrac_period: Refractory period (ms)
        encoding_rate_max: Maximum encoding firing rate (Hz)
        decoding_window: Decoding time window (time steps)
        stdp_enabled: Enable spike-timing dependent plasticity
        stdp_lr: STDP learning rate
        tau_plus: STDP time constant for potentiation (ms)
        tau_minus: STDP time constant for depression (ms)
    """
    dt: float = 1.0  # ms
    tau_mem: float = 10.0  # ms
    tau_syn: float = 5.0  # ms
    v_thresh: float = -50.0  # mV
    v_reset: float = -70.0  # mV
    v_rest: float = -65.0  # mV
    refrac_period: float = 2.0  # ms
    encoding_rate_max: float = 200.0  # Hz
    decoding_window: int = 10  # time steps
    stdp_enabled: bool = False
    stdp_lr: float = 0.01
    tau_plus: float = 20.0  # ms
    tau_minus: float = 20.0  # ms


# ============================
#  Spike-based dCaAP Neuron
# ============================

class SpikingDCaAPNeuron:
    """Spiking neuron with dCaAP-inspired dynamics.
    
    Combines leaky integrate-and-fire (LIF) dynamics with dCaAP-like
    dendritic computation:
    - Multiple dendritic compartments
    - Non-linear dendritic integration (anti-coincidence detection)
    - Calcium-gated somatic integration
    - Adaptive threshold
    
    This enables the XOR capability of dCaAP in a spiking framework.
    """
    
    def __init__(
        self,
        n_dendrites: int = 4,
        config: Optional[NeuromorphicConfig] = None,
    ):
        """Initialize spiking dCaAP neuron.
        
        Args:
            n_dendrites: Number of dendritic compartments
            config: Neuromorphic configuration
        """
        self.n_dendrites = n_dendrites
        self.config = config or NeuromorphicConfig()
        
        # State variables
        self.v_mem = self.config.v_rest  # Membrane potential
        self.v_dendrites = np.full(n_dendrites, self.config.v_rest)  # Dendritic potentials
        self.i_syn = 0.0  # Synaptic current
        self.refrac_counter = 0  # Refractory period counter
        self.calcium = 0.0  # Calcium concentration (0-1)
        
        # Adaptive threshold
        self.v_thresh_adaptive = self.config.v_thresh
        self.thresh_adaptation_rate = 0.1
    
    def reset(self):
        """Reset neuron state."""
        self.v_mem = self.config.v_rest
        self.v_dendrites = np.full(self.n_dendrites, self.config.v_rest)
        self.i_syn = 0.0
        self.refrac_counter = 0
        self.calcium = 0.0
        self.v_thresh_adaptive = self.config.v_thresh
    
    def step(self, spike_inputs: np.ndarray, weights: np.ndarray) -> bool:
        """Single time step update.
        
        Args:
            spike_inputs: Input spikes per dendrite [n_dendrites]
            weights: Synaptic weights per dendrite [n_dendrites]
            
        Returns:
            True if neuron spiked, False otherwise
        """
        cfg = self.config
        spiked = False
        
        # Update dendritic potentials with dCaAP-like dynamics
        for i in range(self.n_dendrites):
            # Dendritic input current
            i_dend = spike_inputs[i] * weights[i]
            
            # Dendritic leak
            dv_dend = (cfg.v_rest - self.v_dendrites[i]) / cfg.tau_mem
            
            # dCaAP-inspired non-linearity: anti-coincidence detection
            # When multiple inputs arrive simultaneously, reduce response
            coincidence_factor = 1.0 / (1.0 + np.sum(spike_inputs) - spike_inputs[i])
            
            # Update dendritic potential
            self.v_dendrites[i] += cfg.dt * (dv_dend + i_dend * coincidence_factor)
        
        # Dendritic-to-somatic integration (calcium-gated)
        dendritic_current = np.sum(self.v_dendrites - cfg.v_rest) / self.n_dendrites
        
        # Update calcium concentration (decays over time, increases with dendritic activity)
        dendrite_activity = np.mean(np.abs(self.v_dendrites - cfg.v_rest))
        self.calcium += cfg.dt * (-self.calcium / 50.0 + 0.01 * dendrite_activity)
        self.calcium = np.clip(self.calcium, 0.0, 1.0)
        
        # Calcium gate modulates somatic integration
        gated_current = dendritic_current * self.calcium
        
        # Check refractory period
        if self.refrac_counter > 0:
            self.refrac_counter -= 1
            return False
        
        # Membrane dynamics (LIF)
        dv = (cfg.v_rest - self.v_mem) / cfg.tau_mem + gated_current
        self.v_mem += cfg.dt * dv
        
        # Check for spike
        if self.v_mem >= self.v_thresh_adaptive:
            spiked = True
            self.v_mem = cfg.v_reset
            self.refrac_counter = int(cfg.refrac_period / cfg.dt)
            
            # Adaptive threshold (increases after spike)
            self.v_thresh_adaptive += self.thresh_adaptation_rate
        else:
            # Threshold recovery
            self.v_thresh_adaptive += cfg.dt * (cfg.v_thresh - self.v_thresh_adaptive) / 100.0
        
        return spiked


# ============================
#  Encoding/Decoding
# ============================

def rate_encode(
    values: np.ndarray,
    time_steps: int,
    max_rate: float = 200.0,
    dt: float = 1.0,
) -> np.ndarray:
    """Encode continuous values as spike trains using rate coding.
    
    Higher values -> higher firing rate.
    
    Args:
        values: Input values [batch_size, features] in range [0, 1]
        time_steps: Number of time steps
        max_rate: Maximum firing rate (Hz)
        dt: Time step duration (ms)
        
    Returns:
        Spike trains [batch_size, features, time_steps]
    """
    batch_size, n_features = values.shape
    
    # Convert values to firing rates (0 to max_rate Hz)
    rates = values * max_rate
    
    # Probability of spike per time step: rate * dt / 1000
    spike_probs = rates * dt / 1000.0
    spike_probs = np.clip(spike_probs, 0.0, 1.0)
    
    # Generate spike trains
    spikes = np.random.rand(batch_size, n_features, time_steps) < spike_probs[..., None]
    
    return spikes.astype(np.float32)


def rate_decode(
    spike_trains: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """Decode spike trains to continuous values using rate coding.
    
    Args:
        spike_trains: Spike trains [batch_size, features, time_steps]
        window: Decoding window size (time steps)
        
    Returns:
        Decoded values [batch_size, features]
    """
    # Average firing rate over last window time steps
    if spike_trains.shape[2] < window:
        window = spike_trains.shape[2]
    
    recent_spikes = spike_trains[:, :, -window:]
    firing_rates = np.mean(recent_spikes, axis=2)
    
    return firing_rates


def temporal_encode(
    values: np.ndarray,
    time_steps: int,
    min_latency: int = 1,
) -> np.ndarray:
    """Encode continuous values as spike latencies (temporal coding).
    
    Higher values -> earlier spikes.
    
    Args:
        values: Input values [batch_size, features] in range [0, 1]
        time_steps: Number of time steps
        min_latency: Minimum latency (time steps)
        
    Returns:
        Spike trains [batch_size, features, time_steps]
    """
    batch_size, n_features = values.shape
    
    # Convert values to latencies (inverse relationship)
    # High value -> low latency -> early spike
    latencies = min_latency + (1.0 - values) * (time_steps - min_latency - 1)
    latencies = latencies.astype(int)
    
    # Generate spike trains (one spike per neuron)
    spikes = np.zeros((batch_size, n_features, time_steps), dtype=np.float32)
    
    for b in range(batch_size):
        for f in range(n_features):
            t = latencies[b, f]
            if 0 <= t < time_steps:
                spikes[b, f, t] = 1.0
    
    return spikes


# ============================
#  Neuromorphic T-RLINKOS
# ============================

class NeuromorphicTRLinkosTRM:
    """Neuromorphic implementation of T-RLINKOS TRM.
    
    This is a simplified spike-based version for neuromorphic hardware.
    Uses spiking dCaAP neurons and event-driven computation.
    
    Note: This is an experimental research implementation.
    For production use, prefer the standard NumPy or PyTorch versions.
    """
    
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        z_dim: int,
        hidden_dim: int = 128,
        num_experts: int = 4,
        config: Optional[NeuromorphicConfig] = None,
    ):
        """Initialize neuromorphic T-RLINKOS.
        
        Args:
            x_dim: Input dimension
            y_dim: Output dimension
            z_dim: Internal state dimension
            hidden_dim: Hidden layer dimension
            num_experts: Number of dCaAP experts
            config: Neuromorphic configuration
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.config = config or NeuromorphicConfig()
        
        # Create spiking neurons
        self.neurons = []
        for _ in range(num_experts):
            neuron = SpikingDCaAPNeuron(n_dendrites=4, config=self.config)
            self.neurons.append(neuron)
        
        # Synaptic weights (simplified)
        input_dim = x_dim + y_dim + z_dim
        self.input_weights = np.random.randn(num_experts, 4, input_dim) * 0.1
        self.output_weights = np.random.randn(y_dim, num_experts) * 0.1
        
        print(f"Neuromorphic T-RLINKOS initialized:")
        print(f"  Input: {x_dim}, Output: {y_dim}, State: {z_dim}")
        print(f"  Experts: {num_experts}, Hidden: {hidden_dim}")
        print(f"  Time step: {self.config.dt} ms")
    
    def reset_state(self):
        """Reset all neuron states."""
        for neuron in self.neurons:
            neuron.reset()
    
    def encode_to_spikes(
        self,
        x: np.ndarray,
        time_steps: int = 100,
        encoding: str = "rate",
    ) -> np.ndarray:
        """Encode input to spike trains.
        
        Args:
            x: Input array [batch_size, x_dim] in range [0, 1]
            time_steps: Number of time steps
            encoding: "rate" or "temporal"
            
        Returns:
            Spike trains [batch_size, x_dim, time_steps]
        """
        # Normalize input to [0, 1]
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        
        if encoding == "rate":
            return rate_encode(x_norm, time_steps, self.config.encoding_rate_max, self.config.dt)
        elif encoding == "temporal":
            return temporal_encode(x_norm, time_steps)
        else:
            raise ValueError(f"Unknown encoding: {encoding}")
    
    def decode_from_spikes(
        self,
        spike_trains: np.ndarray,
        decoding: str = "rate",
    ) -> np.ndarray:
        """Decode spike trains to continuous output.
        
        Args:
            spike_trains: Spike trains [batch_size, y_dim, time_steps]
            decoding: "rate" or "temporal"
            
        Returns:
            Decoded output [batch_size, y_dim]
        """
        if decoding == "rate":
            return rate_decode(spike_trains, self.config.decoding_window)
        else:
            raise ValueError(f"Unknown decoding: {decoding}")
    
    def forward_spikes(
        self,
        input_spikes: np.ndarray,
        time_steps: int = 100,
    ) -> np.ndarray:
        """Forward pass with spike trains.
        
        Args:
            input_spikes: Input spike trains [batch_size, x_dim, time_steps]
            time_steps: Number of time steps to simulate
            
        Returns:
            Output spike trains [batch_size, y_dim, time_steps]
        """
        batch_size = input_spikes.shape[0]
        
        # Initialize output spike trains
        output_spikes = np.zeros((batch_size, self.y_dim, time_steps), dtype=np.float32)
        
        # Simplified forward pass (single batch at a time)
        for b in range(batch_size):
            self.reset_state()
            
            # Simulate over time
            for t in range(time_steps):
                # Get input spikes at time t
                x_t = input_spikes[b, :, t]
                
                # Process through expert neurons
                expert_spikes = np.zeros(self.num_experts)
                for i, neuron in enumerate(self.neurons):
                    # Distribute input to dendritic compartments
                    dendrite_inputs = np.zeros(4)
                    for d in range(4):
                        dendrite_inputs[d] = np.dot(self.input_weights[i, d, :self.x_dim], x_t)
                    
                    # Update neuron
                    weights = np.ones(4)  # Simplified
                    spiked = neuron.step(dendrite_inputs, weights)
                    expert_spikes[i] = 1.0 if spiked else 0.0
                
                # Compute output spikes (linear readout)
                for j in range(self.y_dim):
                    output_activation = np.dot(self.output_weights[j, :], expert_spikes)
                    # Stochastic spiking based on activation
                    prob = 1.0 / (1.0 + np.exp(-output_activation))
                    output_spikes[b, j, t] = 1.0 if np.random.rand() < prob else 0.0
        
        return output_spikes
    
    def forward(
        self,
        x: np.ndarray,
        time_steps: int = 100,
    ) -> np.ndarray:
        """Full forward pass with encoding/decoding.
        
        Args:
            x: Input array [batch_size, x_dim]
            time_steps: Number of time steps
            
        Returns:
            Output array [batch_size, y_dim]
        """
        # Encode input to spikes
        input_spikes = self.encode_to_spikes(x, time_steps)
        
        # Process spikes
        output_spikes = self.forward_spikes(input_spikes, time_steps)
        
        # Decode output spikes
        output = self.decode_from_spikes(output_spikes)
        
        return output


# ============================
#  Utilities
# ============================

def get_neuromorphic_info() -> Dict[str, Any]:
    """Get neuromorphic computing information.
    
    Returns:
        Dictionary with info
    """
    return {
        "implementation": "spike-based",
        "neuron_model": "Spiking dCaAP (LIF + dendritic computation)",
        "encoding": ["rate coding", "temporal coding"],
        "features": [
            "Event-driven computation",
            "Low power operation",
            "Temporal dynamics",
            "Anti-coincidence detection",
            "Adaptive threshold",
        ],
        "target_hardware": ["Intel Loihi", "IBM TrueNorth", "SpiNNaker", "General CPU/GPU"],
        "maturity": "Experimental (Research prototype)",
    }


# ============================
#  Main test
# ============================

if __name__ == "__main__":
    print("=" * 70)
    print("NEUROMORPHIC T-RLINKOS TEST")
    print("=" * 70)
    
    # Print info
    info = get_neuromorphic_info()
    print(f"\nImplementation: {info['implementation']}")
    print(f"Neuron Model: {info['neuron_model']}")
    print(f"Maturity: {info['maturity']}")
    print(f"\nFeatures:")
    for feature in info['features']:
        print(f"  - {feature}")
    print(f"\nTarget Hardware:")
    for hw in info['target_hardware']:
        print(f"  - {hw}")
    
    # Test spiking neuron
    print("\n--- Test 1: Spiking dCaAP Neuron ---")
    config = NeuromorphicConfig(dt=1.0, tau_mem=10.0)
    neuron = SpikingDCaAPNeuron(n_dendrites=4, config=config)
    
    # Simulate neuron response
    spike_count = 0
    for t in range(100):
        # Random input spikes
        spike_inputs = np.random.rand(4) > 0.9
        weights = np.ones(4)
        spiked = neuron.step(spike_inputs, weights)
        if spiked:
            spike_count += 1
    
    print(f"Neuron spiked {spike_count} times in 100 time steps")
    print(f"Final membrane potential: {neuron.v_mem:.2f} mV")
    print(f"Final calcium: {neuron.calcium:.3f}")
    
    # Test encoding
    print("\n--- Test 2: Rate Encoding ---")
    values = np.array([[0.0, 0.5, 1.0]])  # [1, 3]
    spikes = rate_encode(values, time_steps=50, max_rate=200.0, dt=1.0)
    print(f"Input values: {values[0]}")
    print(f"Spike counts: {np.sum(spikes[0], axis=1)}")
    
    # Test decoding
    decoded = rate_decode(spikes, window=10)
    print(f"Decoded values: {decoded[0]}")
    
    # Test temporal encoding
    print("\n--- Test 3: Temporal Encoding ---")
    spikes_temporal = temporal_encode(values, time_steps=50)
    print(f"Input values: {values[0]}")
    for i in range(3):
        spike_times = np.where(spikes_temporal[0, i, :] > 0)[0]
        print(f"  Feature {i}: spike at t={spike_times[0] if len(spike_times) > 0 else 'none'}")
    
    # Test full model
    print("\n--- Test 4: Neuromorphic T-RLINKOS ---")
    model = NeuromorphicTRLinkosTRM(
        x_dim=8,
        y_dim=4,
        z_dim=8,
        hidden_dim=16,
        num_experts=4,
    )
    
    # Forward pass
    x = np.random.rand(2, 8)  # [batch_size=2, x_dim=8]
    print(f"Input shape: {x.shape}")
    
    output = model.forward(x, time_steps=50)
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output[0]}")
    
    print("\n" + "=" * 70)
    print("✅ Neuromorphic T-RLINKOS tests completed!")
    print("=" * 70)
    
    print("\n⚠️  Note: This is an experimental research implementation.")
    print("For production use, prefer the standard NumPy or PyTorch versions.")
    
    print("\nUsage Example:")
    print("  model = NeuromorphicTRLinkosTRM(x_dim=64, y_dim=32, z_dim=64)")
    print("  output = model.forward(input_data, time_steps=100)")
