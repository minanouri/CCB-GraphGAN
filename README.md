# Cycle-Consistent Bidirectional Graph Generative Adversarial Network


## Overview  

This project introduces **CCB-GraphGAN** (Cycle-Consistent Bidirectional Graph Generative Adversarial Network), a model designed to improve the accuracy and timeliness of **lane-level anomaly detection on freeways**. By modeling traffic as graph-structured data, CCB-GraphGAN captures the interdependencies of traffic flows across lanes and leverages them within an adversarial learning framework.

The model extends the Bidirectional GAN (BiGAN) framework with Graph Attention Networks (GATs) and incorporates cycle consistency constraints in both the data and latent spaces. This enables the network to jointly learn the distribution of normal traffic and its latent representations, ensuring reliable reconstruction of normal traffic patterns. During inference, anomalies are identified when reconstruction errors spike in specific nodes, allowing the system to precisely pinpoint disruptions at the node (lane) level.  

Key features of the CCB-GraphGAN model include:  
- **Adversarial graph learning** to capture complex traffic dependencies.  
- **Cycle consistency** for reliable reconstruction of traffic data and latent features.  
- **Autoencoder-based detection** to identify anomalies at the lane level during inference.  


The model is evaluated on the [Freeway Traffic Anomalous Event Detection (FT-AED) dataset](https://acoursey3.github.io/ft-aed/), where it: 
- Demonstrates effectiveness in real-world crash case studies.  
- Outperforms baseline models in early anomaly detection.  
- Reduces detection delay by an average of **5 minutes earlier than official crash reports**.  
- Achieves a **1% false positive rate** with fewer missed crashes.  
 

## Requirements

Install the required dependencies with:

```bash
pip install -r requirements.txt
