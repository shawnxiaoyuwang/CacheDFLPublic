# Cache-DFL

A **cache-based Distributed Federated Learning** framework (experimental) for **"Decentralized Federated Learning with Model Caching on Mobile Agents."**. This repository explores new strategies for caching and data loading in distributed or federated learning scenarios.

## Features

- **Federated Learning Trainer**: A customizable trainer for simulating federated learning across multiple nodes.
- **Caching Mechanism**: Implements a caching module to efficiently manage repeated data access.
- **Data Handling Utilities**: Convenient data loaders for local or remote datasets.
- **Extensible Design**: Easily modify or extend the core classes to fit your own research.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shawnxiaoyuwang/Cache-DFL.git
   cd Cache-DFL
   Install dependencies (if using requirements.txt):
2. **Install dependencies (if using requirements.txt)**:
   ```bash
   pip install -r requirements.txt
   ```
   Or if you prefer a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   
# Project Structure
   ```plaintext
   Cache-DFL/
   ├── cache_dfl/
   │   ├── models/
   │   ├── aggregation.py
   │   ├── Cache_algorithm.py
   │   ├── Cached_DFL_mpi.py
   │   ├── Cached_DFL_single.py
   │   ├── data.py
   │   ├── model.py
   │   ├── road_sim.py
   │   ├── seed_setter.py
   │   ├── utils.py
   │   └── utils_cnn.py
   ├── examples/
   │   └── demo.py
   ├── tests/
   │   ├── test_trainer.py
   │   └── ...
   ├── requirements.txt
   ├── LICENSE
   └── README.md
```
* `cachedfl/`: Core Python package containing the main source code.
* `examples/`: Example scripts that show how to use the code.
* `tests/`: Automated tests using pytest or unittest.
* `requirements.txt`: Python dependencies list.
* `README.md`: This file.
## License

* This project is licensed under the MIT License – see the LICENSE file for details.

## Acknowledgments
* Xiaoyu Wang, New York University
* Oral Presentation at AAAI 2025
* Extended Version:
  Wang, Xiaoyu, Guojun Xiong, Houwei Cao, Jian Li, and Yong Liu. "Decentralized Federated Learning with Model Caching on Mobile Agents." arXiv preprint arXiv:2408.14001 (2024).
  https://arxiv.org/abs/2408.14001
