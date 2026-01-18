<div align="center">

# 🏥 AI-Powered Radiology Workflow Optimizer
### *Intelligent Automation for Modern Healthcare*

[![React](https://img.shields.io/badge/Frontend-React_18-61DAFB?style=for-the-badge&logo=react)](https://reactjs.org/)
[![Node.js](https://img.shields.io/badge/Backend-Node.js_18-339933?style=for-the-badge&logo=node.js)](https://nodejs.org/)
[![Python](https://img.shields.io/badge/ML-Python_3.10-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/AI-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)


[**Explore Docs**](#-documentation) · [**Report Bug**](issues/new) · [**Request Feature**](issues/new)

</div>

---

## 📖 Overview

The **AI-Powered Radiology Workflow Optimizer** is a comprehensive hospital management system designed to streamline the radiology department's operations. By integrating **Machine Learning** with robust **Full-Stack** architecture, it automates patient prioritization, ensures efficient doctor allocation, and accelerates diagnosis through intelligent image analysis.

Our system bridges the gap between traditional hospital workflows and modern AI capabilities, ensuring critical cases are flagged immediately and processed with priority.

---

## ✨ Key Features

### 🧠 **AI & Machine Learning**
- **Smart Patient Prioritization**: NLP-based analysis of symptoms to assign urgency levels (Critical, Urgent, Normal).
- **Automated Image Analysis**: Pre-screening of X-ray and MRI scans using PyTorch models.
- **Auto-Generated Reports**: Instant generation of preliminary radiology reports in PDF format.
- **Advanced Inference**: Utilization of Duoformer and custom CNN architectures for high-accuracy medical text and image understanding.

### ⚡ **Operational Efficiency**
- **Intelligent Queue Management**: **RabbitMQ**-powered asynchronous processing ensures the main application remains responsive while handling heavy ML workloads.
- **Real-time Workflow Tracking**: Live status updates for report generation (Pending → Processing → Completed).
- **Resource Optimization**: Automatic assignment of patients to available doctors based on specialization and current load.

### 💻 **Modern User Experience**
- **Interactive Dashboards**: Role-specific portals for **Admins**, **Doctors**, and **Patients** using **React** and **TailwindCSS**.
- **DICOM Viewer Integration**: Native support for viewing high-resolution medical imaging directly in the browser via **Cornerstone.js**.
- **Secure Authentication**: robust Role-Based Access Control (RBAC) powered by **JWT**.

---

## 🏗️ System Architecture

The project follows a **Microservices-inspired** architecture to ensure scalability, fault tolerance, and separation of concerns.

```mermaid
graph TD
    subgraph Frontend ["🖥️ User Interface"]
        A[React Application]
        A -->|HTTP/REST| B[API Gateway / Backend]
    end

    subgraph Backend ["⚙️ Core Services"]
        B[Node.js + Express Server]
        B -->|Auth & User Data| C[(MongoDB)]
        B -->|Structured Data| D[(MySQL / PostgreSQL)]
        B -->|Cache| E[(Redis)]
        B -->|Job Queue| F[RabbitMQ]
    end

    subgraph ML_Services ["🤖 AI/ML Engine"]
        F -->|Consume Tasks| G[Python Worker Service]
        G -->|Inference| H[PyTorch Models]
        H -->|Results| G
        G -->|Update Status| B
    end
