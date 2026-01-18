# AI-Powered Radiology Workflow Optimiser - Technology Report

## Overview

This project is a full-stack hospital radiology management system with ML-powered patient prioritization. It uses a modern microservices architecture with separate frontend, backend, and ML components.

---

## 🖥️ Frontend Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 18.3.1 | Core UI library for building component-based interfaces |
| **Vite** | 7.2.4 | Fast build tool and dev server for modern web development |
| **TypeScript** | 5.8.3 | Type-safe JavaScript for better developer experience |
| **TailwindCSS** | 3.4.18 | Utility-first CSS framework for rapid styling |
| **Radix UI** | Various | Accessible, unstyled UI component primitives |
| **React Router DOM** | 6.30.1 | Client-side routing for single-page application |
| **React Query** | 5.83.0 | Server state management and data fetching |
| **Recharts** | 2.15.4 | Charts and data visualization |
| **Lucide React** | 0.462.0 | Icon library |
| **Sonner** | 1.7.4 | Toast notifications |
| **Zod** | 3.25.76 | Schema validation |
| **React Hook Form** | 7.61.1 | Form state management |

### DICOM Viewing
| Technology | Purpose |
|------------|---------|
| **Cornerstone Core** | Medical image rendering engine |
| **Cornerstone WADO Image Loader** | DICOM image loading |
| **dicom-parser** | DICOM file parsing |

---

## ⚙️ Backend Technologies

### Core Framework
| Technology | Version | Purpose |
|------------|---------|---------|
| **Node.js** | - | JavaScript runtime for server-side code |
| **Express.js** | 5.1.0 | Web framework for REST API |

### Databases
| Technology | Version | Purpose |
|------------|---------|---------|
| **PostgreSQL** | - | Primary relational database for patients, doctors, prescriptions, DICOM metadata |
| **Sequelize** | 6.35.2 | ORM for PostgreSQL database operations |
| **MongoDB** | - | NoSQL database for user authentication and login activity |
| **Mongoose** | 9.0.0 | ODM for MongoDB operations |

### Message Queue
| Technology | Version | Purpose |
|------------|---------|---------|
| **RabbitMQ** | - | Message broker for async ML processing |
| **amqplib** | 0.10.9 | Node.js AMQP client for RabbitMQ |

### Authentication & Security
| Technology | Version | Purpose |
|------------|---------|---------|
| **JWT** | 9.0.2 | JSON Web Tokens for authentication |
| **bcryptjs** | 3.0.3 | Password hashing |
| **cookie-parser** | 1.4.7 | Cookie handling for sessions |
| **CORS** | 2.8.5 | Cross-origin resource sharing |

### File Handling
| Technology | Version | Purpose |
|------------|---------|---------|
| **Multer** | 2.0.2 | File upload middleware |
| **dicom-parser** | 1.8.21 | DICOM file parsing on backend |

### Utilities
| Technology | Version | Purpose |
|------------|---------|---------|
| **Winston** | 3.18.3 | Structured logging |
| **Axios** | 1.13.2 | HTTP client for external requests |
| **UUID** | 9.0.0 | Unique identifier generation |
| **dotenv** | 16.3.1 | Environment variable management |

### Monitoring
| Technology | Version | Purpose |
|------------|---------|---------|
| **prom-client** | 15.1.3 | Prometheus metrics for monitoring |

---

## 🤖 Machine Learning Technologies

### Core ML Framework
| Technology | Version | Purpose |
|------------|---------|---------|
| **PyTorch** | ≥2.0.0 | Deep learning framework for priority classifier |
| **scikit-learn** | ≥1.3.0 | Traditional ML algorithms and train/test split |
| **NumPy** | ≥1.24.0 | Numerical computing |
| **Pandas** | ≥2.0.0 | Data manipulation and CSV handling |

### ML API & Worker
| Technology | Version | Purpose |
|------------|---------|---------|
| **FastAPI** | ≥0.100.0 | High-performance async API for ML endpoints |
| **Uvicorn** | ≥0.23.0 | ASGI server for FastAPI |
| **Pika** | ≥1.3.0 | Python AMQP client for RabbitMQ worker |

### Document Processing
| Technology | Version | Purpose |
|------------|---------|---------|
| **PyPDF2** | ≥3.0.0 | PDF text extraction for medical history |
| **ReportLab** | ≥4.0.0 | PDF report generation |
| **Pillow** | ≥10.0.0 | Image processing |

### Monitoring
| Technology | Version | Purpose |
|------------|---------|---------|
| **prometheus_client** | ≥0.17.0 | Python Prometheus metrics |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React + Vite)                  │
│  • Admin Dashboard    • Doctor Dashboard    • Patient Portal     │
│  • DICOM Viewer       • Activity Logs       • Patient Management │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Backend (Express.js)                         │
│  • REST API          • Authentication       • File Uploads       │
│  • Activity Logging  • Patient CRUD         • Doctor Management  │
└─────────────────────────────────────────────────────────────────┘
                │                               │
        ┌───────┴───────┐               ┌───────┴───────┐
        ▼               ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  PostgreSQL  │ │   MongoDB    │ │   RabbitMQ   │ │  ML Worker   │
│  (Sequelize) │ │  (Mongoose)  │ │   (AMQP)     │ │  (PyTorch)   │
│              │ │              │ │              │ │              │
│ • Patients   │ │ • Users      │ │ • priority_  │ │ • Priority   │
│ • Doctors    │ │ • Sessions   │ │   queue      │ │   Classifier │
│ • DICOM      │ │ • Login      │ │ • cases_out  │ │ • Rule-based │
│ • Reports    │ │   Activity   │ │ • cases_dead │ │   Fallback   │
│ • Logs       │ │              │ │              │ │              │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

---

## 📁 Key File Locations

| Component | Location |
|-----------|----------|
| Frontend | `RadiologyFrontend/` |
| Backend Server | `integrated-backend/server.js` |
| Database Models | `integrated-backend/models/` |
| ML Priority System | `integrated-backend/prioritization-ml/` |
| API Routes | `integrated-backend/routes/` |

---

## 🚀 How to Run

```bash
# Backend
cd integrated-backend
npm install
npm start

# ML Worker
npm run start:ml-priority

# Frontend
cd RadiologyFrontend
npm install
npm run dev
```

**Required Services**: PostgreSQL, MongoDB, RabbitMQ
