Data Collection (Slides 2–4)

    Slide 2: Data Collection – Definition
        Definition: Data collection in MLOps refers to the process of gathering and accessing data from multiple sources in a standardized format for ML model development.
    Slide 3: Data Collection – Accessibility
        Definition: Ensuring that data required for model training and testing is easily accessible to data scientists and engineers within a few minutes.
        Tool Examples:
            Google BigQuery, AWS S3, Databricks Lakehouse
    Slide 4: Data Collection – Data Lake/Warehouse
        Definition: Central repositories, such as data lakes and data warehouses, facilitate the consolidation and storage of structured and unstructured data for ML workflows.
        Tool Examples:
            Apache Hadoop, Amazon S3, Snowflake

Feature Extraction Process (Slides 5–8)

    Slide 5: Feature Extraction – Definition
        Definition: Feature extraction in MLOps is the process of transforming raw data into structured features that are directly usable by ML models.

    Slide 6: Data Wrangling
        Definition: The process of transforming and cleaning raw data into structured formats suitable for modeling.
        Tool Examples:
            Pandas, Dask, Trifacta

    Slide 7: Data Processing Pipelines
        Definition: Implementing automated data processing pipelines ensures that data flows from source to model in a structured and reproducible manner.
        Tool Examples:
            Apache Airflow, AWS Glue

    Slide 8: Feature Store Usage
        Definition: Utilizing a feature store allows the central management, sharing, and reuse of model features across projects.
        Tool Examples:
            Feast, AWS SageMaker Feature Store

Git Strategy (Slides 9–10)

    Slide 9: Git Strategy – Definition
        Definition: Git strategy involves version control practices that allow structured development and collaboration within ML projects.

    Slide 10: Git Branching & Versioning
        Definition: Establishing a Git branching strategy and versioning system to track code changes, manage environments, and control production deployments.
        Tool Examples:
            GitHub, GitLab, Bitbucket

Notebook & Code Strategy (Slides 11–13)

    Slide 11: Notebook & Code Strategy – Definition
        Definition: This capability encompasses standardized practices for exploratory data analysis (EDA), code versioning, and experiment tracking within ML projects.

    Slide 12: Notebook Use for EDA
        Definition: Interactive notebooks are employed for conducting Exploratory Data Analysis (EDA), providing a flexible and visual approach to data exploration.
        Tool Examples:
            Jupyter Notebook, Google Colab

    Slide 13: Code IDEs for Refactoring
        Definition: Integrated Development Environments (IDEs) like PyCharm or VS Code are used to refactor code and build robust ML pipelines.
        Tool Examples:
            PyCharm, VS Code, Atom

Project, Model, and Technical Documentation (Slides 14–17)

    Slide 14: Documentation – Definition
        Definition: Documentation in MLOps involves systematically recording every stage of the model lifecycle, from data processing to model deployment, ensuring reproducibility and auditability.

    Slide 15: Data Dictionary Documentation
        Definition: A data dictionary documents all features used in a model, including their descriptions and intended purposes.
        Tool Examples:
            Confluence, GitHub README files, Excel Sheets

    Slide 16: Experiment Tracking
        Definition: Experiment tracking allows the logging of all experiments, hyperparameters, datasets, and outcomes for easy comparison and reproducibility.
        Tool Examples:
            MLflow, DVC, Weights & Biases

    Slide 17: Release Notes & Versioning
        Definition: Release notes document changes made to the model, code, and data across versions to track model evolution.
        Tool Examples:
            Git tags, JIRA, Azure DevOps


  =====================


  AI Governance (Slides 1–4)

Slide 1: AI Governance – Definition

    Definition: This capability ensures that AI models are developed, tested, and deployed with ethical, transparent, and regulatory-compliant practices.

Slide 2: Data & Model Bias Implementation and Verification

    Definition: Mechanisms to detect and mitigate bias in data and models are implemented and routinely verified, ensuring fairness in AI outputs.
    Tool Examples: Fairness Indicators, IBM AI Fairness 360, What-If Tool

Slide 3: Explainability Tests

    Definition: Ensures model predictions are interpretable to users and stakeholders, increasing trust and enabling compliance in regulated sectors.
    Tool Examples: SHAP, LIME, IBM AI Explainability 360

Slide 4: AI Product Governance Validation

    Definition: Validates that the AI product complies with governance requirements like data privacy, transparency, and fairness throughout the lifecycle.
    Tool Examples: IBM OpenScale, AI Explainability 360

Code Quality Gates (Slides 5–8)

Slide 5: Code Quality Gates – Definition

    Definition: This capability includes checks and standards to maintain high-quality, efficient, and error-free code.

Slide 6: Code Quality Standards (DRY, YAGNI, KISS)

    Definition: Adheres to principles such as DRY (Don’t Repeat Yourself), YAGNI (You Aren't Gonna Need It), and KISS (Keep It Simple, Stupid) for efficient and maintainable code.
    Tool Examples: Pylint, SonarQube, Black

Slide 7: Pre-Commit Hooks

    Definition: Automated checks that run before committing code, enforcing coding standards and preventing quality issues from entering the codebase.
    Tool Examples: pre-commit framework, Husky, Lint-Staged

Slide 8: Pull Request Review for Quality Assurance

    Definition: Requires pull request reviews to confirm the inclusion of essential tests and high-quality code before integration.
    Tool Examples: GitHub Code Review, GitLab, Bitbucket

Data Quality Gates (Slides 9–12)

Slide 9: Data Quality Gates – Definition

    Definition: Ensures data used in ML models meets quality standards for consistency, validity, and accuracy.

Slide 10: Data Quality Principles Adherence

    Definition: Maintains principles of data quality—accuracy, completeness, consistency, timeliness, validity, and uniqueness—to support reliable ML outputs.
    Tool Examples: Great Expectations, Deequ

Slide 11: Fairness Tests

    Definition: Assesses model predictions for fairness, ensuring equitable outcomes across all user groups and preventing unintended bias.
    Tool Examples: Fairlearn, TensorFlow Model Analysis

Slide 12: Cost of Each Feature Testing

    Definition: Assesses each feature’s computational cost (e.g., latency, RAM usage) during inference to prioritize features with high predictive benefit and low resource demand.
    Tool Examples: SHAP, Py-Spy, TensorFlow Profiler

FinOps (Slides 13–15)

Slide 13: FinOps – Definition

    Definition: Focuses on monitoring, analyzing, and controlling financial costs associated with model training, inference, storage, and experimentation.

Slide 14: Model Training Costs Monitoring and Analysis

    Definition: Tracks and analyzes expenses related to model training, ensuring resource usage remains within budget.
    Tool Examples: AWS CloudWatch, Azure Cost Management, Prometheus

Slide 15: Inference and Storage Costs Monitoring

    Definition: Monitors and optimizes costs associated with model inference and storage, ensuring efficient use of resources.
    Tool Examples: Prometheus, Datadog, Grafana

Monitoring Business Performance (Slides 16–18)

Slide 16: Monitoring Business Performance – Definition

    Definition: Measures and tracks the impact of ML models on key business metrics to ensure alignment with business goals.

Slide 17: Business KPIs Monitoring

    Definition: Tracks and reports key business KPIs related to model performance to gauge the model’s impact on business outcomes.
    Tool Examples: Datadog, Prometheus, Grafana

Slide 18: Continuous Business Outcome Measurement

    Definition: Continuously measures the impact of models on business outcomes, adjusting as needed to maintain alignment with business objectives.
    Tool Examples: Power BI, Tableau, Grafana

Reliable CI/CD Pipelines (Slides 19–21)

Slide 19: Reliable CI/CD Pipelines – Definition

    Definition: Ensures robust, consistent deployment processes across environments with minimal manual intervention.

Slide 20: Automated Test Gates for Code, Data, and Model

    Definition: Establishes automated checks for code, data validation, and model evaluation to maintain standards in deployment.
    Tool Examples: MLflow, PyTest, TFX

Slide 21: Automatic Deployment on Merged Pull Requests

    Definition: Automatically deploys code changes through continuous deployment pipelines when pull requests are merged, ensuring a consistent process.
    Tool Examples: Jenkins, GitLab CI/CD, ArgoCD

Security (Slides 22–26)

Slide 22: Security – Definition

    Definition: Encompasses practices and tools to protect ML models, data, and infrastructure from security risks and unauthorized access.

Slide 23: Automation & Penetration Testing

    Definition: Simulates real-world attacks to test the system’s defenses and ensure its resilience against vulnerabilities.
    Tool Examples: OWASP ZAP, Burp Suite

Slide 24: Security Tests

    Definition: Evaluates model systems for potential vulnerabilities, protecting them from unauthorized access and data breaches.
    Tool Examples: SonarQube, Veracode, Black Duck

Slide 25: Docker Image and Package Scanning

    Definition: Scans Docker images and packages for vulnerabilities, ensuring secure containerized environments and up-to-date libraries.
    Tool Examples: Clair, Anchore, Trivy

Slide 26: Authentication Security Tokens

    Definition: Uses authentication tokens to control access to models and APIs in production, ensuring secure system interactions.
    Tool Examples: JWT, OAuth, Auth0

    ===========================
    Slide 1-2: Repeatable Principles Overview

    Title: Repeatable MLOps Capabilities
    Content: Definition and importance of Repeatable principles in MLOps, enabling reliable, consistent, and reproducible processes across ML projects.

Slide 3: Artefact & Packages – Definition

    Definition: Manages storage, versioning, and packaging of model artifacts, dependencies, and code to support reusable deployments.
    Importance: Ensures that artifacts are consistently available across environments.
    Tool Examples: Artifactory, Azure DevOps, S3 Buckets

Slide 4: Artefact Repository for Code & Packages

    Definition: Stores and versions code, models, and package artifacts to maintain consistency.
    Tool Examples: GitHub Packages, JFrog Artifactory

Slide 5: Deployment Strategy – Definition

    Definition: Standardizes the deployment of models across environments, ensuring repeatability and minimal disruption.
    Importance: Ensures that all deployed models follow a uniform deployment strategy.
    Tool Examples: Kubeflow, SageMaker, Azure ML Pipelines

Slide 6: Real-Time Inference and Batch Deployment

    Definition: Supports both real-time and batch model inference based on project needs.
    Tool Examples: NVIDIA Triton, Apache Spark

Slide 7: Feedback Loop – Definition

    Definition: Collects and integrates feedback data to evaluate model performance and retrain models based on real-world outcomes.
    Importance: Ensures models are responsive to new data and maintain performance over time.
    Tool Examples: Prometheus, Grafana, Apache Kafka

Slide 8: Automated Feedback Data Storage

    Definition: Automatically stores model inputs, outputs, and feedback for continual model assessment and retraining.
    Tool Examples: AWS Kinesis, Azure Event Hub

Slide 9: Infrastructure as Code (IaC) – Definition

    Definition: Manages environment configurations and infrastructure as code for consistent deployments across environments.
    Importance: Enables automated, reproducible, and consistent setup of ML infrastructure.
    Tool Examples: Terraform, Ansible, AWS CloudFormation

Slide 10: IaC and Environment Consistency

    Definition: Standardizes infrastructure configuration across environments to eliminate inconsistencies.
    Tool Examples: HashiCorp Vault, AWS CodePipeline

Slide 11: Maintenance – Definition

    Definition: Implements regular updates, library upgrades, and security checks to maintain system reliability.
    Importance: Ensures that all dependencies and tools remain updated and secure.
    Tool Examples: Dependabot, Azure DevOps Pipelines

Slide 12: Dependency and Library Upgrades

    Definition: Regularly updates ML libraries and dependencies to prevent obsolescence.
    Tool Examples: Pip, Conda

Slide 13: Multi-Environment Management – Definition

    Definition: Manages development, testing, and production environments to ensure reliable testing and deployment.
    Importance: Provides an isolated environment to validate models before production.
    Tool Examples: Docker, Kubernetes, Jenkins

Slide 14: Environment Mirroring and Testing

    Definition: Mirrors production environments in pre-production for thorough testing.
    Tool Examples: Docker Compose, Helm Charts

Slide 15: Repeatable CI/CD Pipelines – Definition

    Definition: Establishes automated CI/CD pipelines to ensure that code, data, and model updates are tested and deployed consistently.
    Importance: Maintains seamless updates and deployment, reducing manual errors.
    Tool Examples: Jenkins, GitLab CI/CD, Azure DevOps

Slide 16: Automated Testing and Rollback Pipelines

    Definition: Incorporates automated testing and rollback in CI/CD to handle failed deployments.
    Tool Examples: Argo CD, GitHub Actions

Slide 17: Traceability & Reproducibility – Definition

    Definition: Provides comprehensive tracking of model development, ensuring each step is traceable and repeatable.
    Importance: Maintains an audit trail and enables troubleshooting.
    Tool Examples: MLflow, DVC, Apache Airflow

Slide 18: Model and Data Lineage Tracking

    Definition: Tracks data sources, model versions, and experiment details.
    Tool Examples: Pachyderm, ClearML

Slide 19: Versioning – Definition

    Definition: Maintains version control for data, code, and models, enabling reproducibility and collaboration.
    Importance: Ensures consistent access to historical versions for reproducibility and auditability.
    Tool Examples: Git, DVC, Azure Repos

Slide 20: Code and Model Versioning

    Definition: Tracks versions of code, models, and configurations for reliable reproduction.
    Tool Examples: Git, MLflow

Closing Slide: Summary of Repeatable Principles

    Title: Summary of Repeatable Principles in MLOps
    Content: Recap of each Repeatable principle, emphasizing their role in achieving reliable and consistent machine learning workflows across environments.
    =======================
Slide 2: Development Templates – Definition

    Definition: Development Templates standardize processes for creating and managing ML projects, ensuring consistent structure and best practices are followed across projects.
    Importance: Promotes efficiency and consistency in project setup, saving time and reducing errors.
    Tool Examples: Cookiecutter, Yeoman, Custom Templates

Slide 3: Data Preparation Templates

    Definition: Templates for data preparation workflows, ensuring standardized data cleaning, transformation, and preprocessing.
    Tool Examples: Pandas, PySpark, Data Wrangling Scripts

Slide 4: Model Training and Tracking Templates

    Definition: Provides standardized templates for model training processes and experiment tracking.
    Tool Examples: MLflow, Kubeflow Pipelines, TensorFlow Extended (TFX)

Slide 5: CI/CD Templates – Definition

    Definition: CI/CD Templates provide standardized workflows for Continuous Integration and Continuous Deployment, automating testing, validation, and model deployment processes.
    Importance: Ensures a streamlined, consistent CI/CD process for every project, allowing efficient, automated model updates and deployments.
    Tool Examples: Jenkins, GitLab CI/CD, Azure Pipelines

Slide 6: Testing and Validation Templates

    Definition: Templates for unit tests, integration tests, and model performance tests ensure code quality, reliability, and reproducibility.
    Tool Examples: PyTest, Unittest, GitHub Actions

Slide 7: Deployment and Rollback Templates

    Definition: Standardized templates for packaging models, managing dependencies, and automating rollback processes.
    Tool Examples: Docker, Helm Charts, Terraform

Slide 8: Advanced Automation – Definition

    Definition: Advanced Automation incorporates automation across the ML lifecycle, reducing manual work and streamlining repeatable tasks, from retraining to monitoring.
    Importance: Increases productivity, supports scalability, and minimizes human error.
    Tool Examples: Apache Airflow, Prefect, Kubeflow

Slide 9: Automated API Creation

    Definition: Automates the creation and management of APIs, enabling ML models to be deployed and accessed through standardized interfaces.
    Tool Examples: FastAPI, Flask, TensorFlow Serving

Slide 10: Automated Model Retraining and Monitoring

    Definition: Automates model retraining based on performance or drift triggers, ensuring the model remains accurate and current.
    Tool Examples: MLflow, Kubeflow Pipelines, SageMaker

Closing Slide: Scalable Principles Summary

    Title: Summary of Scalable MLOps Principles
    Content: Recap of Development Templates, CI/CD Templates, and Advanced Automation, emphasizing the benefits of standardization, automation, and scalability.
    
