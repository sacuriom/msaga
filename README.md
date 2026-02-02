# M-SAGA: Automated LoRaWAN Network Planning in Complex Urban Scenarios

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MATLAB](https://img.shields.io/badge/Language-MATLAB-orange.svg)](https://www.mathworks.com/)
[![Status](https://img.shields.io/badge/Status-Research_Prototype-blue.svg)]()

**A Digital Twin-Based Memetic Approach for Smart City Infrastructure Optimization.**

This repository contains the source code and benchmarking tools associated with the research paper: *"Automated LoRaWAN Network Planning in Complex Urban Scenarios: A Digital Twin-Based Memetic Approach"*, submitted to MDPI Sensors (2025).

## 🚀 Overview

The **Memetic Spatially-Aware Genetic Algorithm (M-SAGA)** is a novel optimization framework designed to solve the Gateway Placement Problem (GPP) in realistic urban environments. Unlike theoretical models based on hexagonal grids, M-SAGA utilizes vector-based OpenStreetMap (OSM) data to act as a "Digital Twin", ensuring that gateways are placed in valid, accessible road intersections.

This suite includes a **Global Dashboard** capable of benchmarking M-SAGA against traditional algorithms (Greedy, B-PSO) using Monte Carlo simulations ($N=50$) in real-time.

### Key Features
* **Vector-Based Digital Twin:** Automatic extraction of urban topology from OSM.
* **Multi-City Scalability:** Validated in diverse topologies (Quito, New York, Madrid, Guayaquil).
* **Advanced KPIs:** Calculates Effective Coverage ($R_{cov}$), Infrastructure Efficiency ($\eta_{eff}$), and Robustness ($I_{rob}$).
* **Interactive Dashboard:** MATLAB App Designer interface for easy visualization.

## 📋 Requirements

* **MATLAB** (R2021a or newer recommended).
* **Statistics and Machine Learning Toolbox**.
* **Internet Connection** (Required to fetch OSM data and Google Maps tiles).
* **Google Maps Static API Key** (Required for background visualization).

## ⚙️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/tu-usuario/M-SAGA-LoRaWAN.git](https://github.com/tu-usuario/M-SAGA-LoRaWAN.git)
    cd M-SAGA-LoRaWAN
    ```

2.  **Setup API Key:**
    Open `src/MDPI14_Benchmark.m` and locate the configuration section. Replace the placeholder with your own Google Maps API Key:
    ```matlab
    cfg.apiKey = 'YOUR_API_KEY_HERE';
    ```
    *(Note: The algorithm works without the background image if the key fails, but visualization is recommended).*

3.  **Run the Benchmark:**
    Open MATLAB, navigate to the `src` folder, and execute:
    ```matlab
    MDPI14_Benchmark
    ```

4.  **Select a City:**
    Choose a target city from the dropdown menu (e.g., *Guayaquil, ECU*) and click **"INICIAR BENCHMARK"**. The system will run 50 iterations per algorithm and display the comparative table.

## 📊 Comparison Algorithms

This repository implements three algorithms for comparative analysis:
1.  **Greedy Algorithm:** A constructive heuristic that maximizes immediate coverage gain.
2.  **B-PSO (Binary Particle Swarm Optimization):** A standard swarm intelligence metaheuristic.
3.  **M-SAGA (Proposed):** A hybrid genetic algorithm with density-based repair operators (Attraction/Repulsion).

## 📜 Citation

If you use this code or methodology in your research, please cite our paper:

```bibtex
@article{acurio2025msaga,
  title={Automated LoRaWAN Network Planning in Complex Urban Scenarios: A Digital Twin-Based Memetic Approach},
  author={Acurio-Maldonado, Santiago and Sacoto-Cabrera, Erwin J. and Meneses, Edison and Huerta, Monica-Karel},
  journal={Sensors (Submitted)},
  year={2025},
  publisher={MDPI}
}
