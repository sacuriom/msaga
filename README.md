# 3D-M-SAGA: Native 3D Optimization of LoRaWAN Network Coverage

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MATLAB](https://img.shields.io/badge/Language-MATLAB-orange.svg)](https://www.mathworks.com/)
[![Status](https://img.shields.io/badge/Status-Academic_Release-blue.svg)]()

**A Native 3D Digital Twin and Memetic Approach for Smart City Infrastructure Optimization.**

This repository contains the source code, Digital Twin extraction tools, advanced benchmarking suites, and 3D simulation assets associated with the research paper: *"3D Digital-Twin–Driven LoRaWAN Gateway Placement Using Memetic Optimization and K-Coverage Network-Health Metrics"*, submitted to the **Future Internet (MDPI) Special Issue: Digital Twins in Next-Generation IoT Networks**.

## 🚀 Overview

The **Native 3D Memetic Spatially-Aware Genetic Algorithm (3D-M-SAGA)** represents a paradigm shift in Low-Power Wide-Area Network (LPWAN) planning. Moving beyond traditional 2D isotropic grids that systematically overestimate coverage, this framework generates a site-specific **3D Morphological Digital Twin**. It fuses vector-based OpenStreetMap (OSM) topologies with NASA SRTM 30m elevation data and an autonomous urban clutter classifier.

By pre-computing deterministic RF link budgets (accounting for ITU-R Knife-Edge diffraction and dielectric absorption), the algorithm optimizes Capital Expenditure (CAPEX) while guaranteeing mission-critical network health and fault tolerance ($K \ge 2$). 

🌍 **Global Scalability:** The framework has been successfully cross-validated in diverse and highly complex topographies (e.g., the longitudinal canyons of Quito, the inter-Andean plateau of Cuenca, and the irregular basin of Ambato). **It can be deployed in any city worldwide** simply by inputting the target Latitude and Longitude bounding box.

## 🌟 Key Features

* 🗺️ **Global Open-Data Integration:** No proprietary 3D models required. Just input the geographic coordinates (Lat/Lon), and the framework autonomously fetches the civic intersections (OSM) and topographic altitude (NASA API).
* 🏙️ **Morphological Clutter Engine:** Autonomous classification of urban density (Dense Urban, Residential, Park/Periphery) for accurate path-loss exponent ($n$) assignment.
* 📡 **ITU-R Physical Constraints:** Integrates hardware specifications (Antenna Height, Tx Power, Gain) and Quality of Service (QoS) mapping based on LoRaWAN **Spreading Factors (SF7-SF12)**.
* 🧬 **Vectorized Memetic Operator:** The "Smart Repair" algorithm uses heuristic attraction/repulsion to prune redundant CAPEX while filling physical coverage holes in real-time.
* 📈 **Rigorous Validation Suites:** Includes parametric grid searches across multiple heights and SFs, alongside Monte Carlo stochastic benchmarking for statistical confidence.
* 💾 **Academic Export Suite:** Automatically generates 600 DPI High-Resolution 3D PNGs, interactive MATLAB `.fig` files, detailed CSV topology logs, and comprehensive HTML deployment reports.

## 📂 Repository Contents

### 📜 Source Code (MATLAB Scripts)
* `MDPI_Suite_MesoCellular_Academic.m`: The core MATLAB suite containing the Digital Twin builder, RF propagation engine, optimization algorithms, and 3D rendering module.
* `MDPI_Sensitivity_Benchmark.m`: The Grid Search suite. Pre-computes Topographic Diffraction once per height and evaluates 36 physical/RF permutations automatically.
* `MDPI_MonteCarlo_Benchmark.m`: The statistical suite. Runs $N=50$ stochastic iterations per algorithm to evaluate variance and mathematical stability.

### 📊 Cross-City Validation Datasets (Pre-computed Results)
As detailed in the manuscript, the repository includes full execution logs and datasets for three highly diverse urban morphologies:
* 📁 `Results_Meso_Ambato_ECU_SF11_...` *(Baseline Scenario - Irregular Basin)*
* 📁 `Results_Meso_Cuenca_ECU_SF11_...` *(Moderate Complexity - Andean Plateau)*
* 📁 `Results_Meso_Quito_ECU_SF11_...` *(High Complexity - Longitudinal Valley)*

**Inside each city folder, you will find:**
  * `Academic_Deployment_Report.html`: Detailed QoS breakdown by Spreading Factor, K-Coverage metrics, and execution logs.
  * `DigitalTwin_[Algorithm].csv`: Raw spatial and RF metadata for every intersection (ready for QGIS/ArcGIS).
  * `3D_Map_[Algorithm].png`: 600 DPI static renderings for paper publication.
  * `Native_Fig_[Algorithm].fig`: Interactive 3D MATLAB figures preserving DataTips.

### 📈 Benchmarking Exports
* `/Benchmark_[City]_[Timestamp]/`: Outputs from the Sensitivity Sweep, including `Global_Sensitivity_Benchmark_Report.html` (evaluating heights of 5m, 10m, 15m and SFs 10, 11, 12).
* `/MONTECARLO_[City]_[Timestamp]/`: Outputs from the stochastic benchmark, including Boxplot PNGs and the `MonteCarlo_Statistical_Report.html` proving the variance reduction of the memetic operator.

## 📋 Requirements

* **MATLAB** (R2021a or newer recommended).
* **Statistics and Machine Learning Toolbox** (Required for K-Medoids clustering).
* **Active Internet Connection** (Strictly required to fetch live OSM and NASA OpenTopoData APIs).

## ⚙️ Installation & Quick Start

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/sacuriom/msaga.git](https://github.com/sacuriom/msaga.git)
   cd msaga
