# DRL-based Split Computing with Cognitive Framework Integration

A Deep Reinforcement Learning solution for dynamic neural network inference splitting between user equipment (UE) and  edge servers, integrated with the VERGE Cognitive Framework for comprehensive experiment tracking and model deployment.

## 🎯 Overview

Split computing optimizes the distribution of deep neural network computations between UEs and edge servers by dynamically determining optimal split points based on current device resources, network conditions, and application requirements. This implementation uses DRL to make intelligent split decisions that minimize inference time, reduce energy consumption, and prevent SLA violations.

## ✨ Key Features

- **Deep Reinforcement Learning**: Actor-Critic architecture for optimal split point selection
- **Multi-objective Optimization**: Balances inference time, energy efficiency, and service reliability
- **Real-time Adaptation**: Continuously adapts to changing network and device conditions
- **Comprehensive Simulation**: Realistic environment modeling with 3GPP-compliant channel data
- **Cognitive Framework Integration**: Full MLOps support with experiment tracking and model deployment
- **Production-Ready**: REST API endpoints for real-world deployment
- **Flexible Architecture**: Supports various DNN models and device types

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.8
pytorch >= 1.9.0
numpy
pandas
matplotlib
cognitive-framework
```

### Installation

```bash
git clone https://github.com/verge-project/split-computing-cf.git
cd split-computing-cf
pip install -r requirements.txt
```

### Basic Usage

```python
from split_computing import SplitComputingAgent, SplitComputingEnv
import cognitive_framework as cf

# Initialize environment with network data
env = SplitComputingEnv(channel_file="channel_data.csv", num_ues=10)

# Create and train DRL agent
agent = SplitComputingAgent(
    state_dim=env.observation_space,
    action_dim=env.action_space,
    max_action=env.action_space - 1
)

# Train with CF integration
cf.set_experiment(experiment_name="Split Computing DRL")
with cf.start_run(run_name="split_computing_training"):
    trained_agent = train_split_computing_agent(env, agent)
```


## 🔧 Configuration

### Training Configuration

```yaml
# training_config.yaml
agent:
  state_dim: 7
  action_dim: 5
  max_action: 4
  discount: 0.99
  tau: 0.005
  policy_noise: 0.2
  noise_clip: 0.5
  policy_freq: 2

training:
  max_timesteps: 1000000
  start_timesteps: 25000
  eval_freq: 5000
  batch_size: 256

environment:
  num_ues: 10
  channel_file: "data/channel_data.csv"
  device_types: ["smartphone", "tablet", "laptop", "iot"]
```

### Environment Setup

```python
# Example UE state representation
ue_state = {
    'cpu_load': 0.45,      # CPU utilization (0-1)
    'gpu_load': 0.3,       # GPU utilization (0-1)  
    'mem_load': 0.6,       # Memory utilization (0-1)
    'bat_level': 0.7,      # Battery level (0-1)
    'rsrp': -95,          # Reference Signal Received Power (dBm)
    'rsrq': -12,          # Reference Signal Received Quality (dB)
    'sinr': 7             # Signal-to-Interference-plus-Noise Ratio (dB)
}
```

## 📊 Monitoring and Visualization

The integration with Cognitive Framework provides comprehensive monitoring:

- **Real-time Training Metrics**: Loss, rewards, energy consumption
- **TensorBoard Integration**: Detailed training visualizations
- **Model Comparison**: Compare different training runs and hyperparameters
- **Performance Dashboards**: Track inference time, reliability, and efficiency

## 🚀 Deployment

### Model Registration

```python
# Register trained model
model_info = cf.pyfunc.log_model(
    artifact_path="split_computing_model",
    python_model=trained_agent,
    artifacts=model_artifacts,
    pip_requirements=[],
    input_example=example_state,
    signature=cf.models.infer_signature(example_state, example_action)
)
```

### Production Deployment

```python
# Deploy to production endpoint
deployment = cf.ModelServing.deploy(
    serving_config=serving_config,
    endpoint_name="split-computing-prod",
    description="Production split computing model",
    tags={"use_case": "edge_optimization", "version": "1.0.0"}
)
```

### Client Usage

```python
# Use deployed model
client = cf.InferenceClient(
    endpoint_url=deployment.endpoint_url,
    auth_token="your_auth_token"
)

result = client.predict(ue_state)
split_point = int(result)
```
t

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{pujol2024dynamic,
  title={Dynamic Split Points Selection in DNNs Inference for Enhanced Edge Computing Performance},
  author={Pujol-Roig, Joan and Kolawole, Oluwatayo Yetunde and Tassi, Andrea and Warren, Daniel},
  journal={VERGE Project},
  year={2024}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work was supported by the European Community through the VERGE project (grant no. 101096034) within the HORIZON-JU-SNS-2022-STREAM-A-01-05 research and innovation program.


---

**Note**: This implementation is part of the VERGE project's open-source contributions to advance edge computing research and facilitate technology transfer to industry.
