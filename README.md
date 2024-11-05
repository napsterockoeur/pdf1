

**1. Background**  
   - **Customer Goals**: Insurance providers aim to enhance the speed, precision, and personalization of pricing models. They need a system that efficiently integrates actuarial expertise with advanced ML capabilities to allow quicker adjustments to tariffs and policies, enabling competitive and fair pricing.
   - **Customer Pains**: Legacy systems lack flexibility and cannot efficiently incorporate ML models, leading to high turnaround times, manual errors, and dependency on IT. There’s a need to reduce the "ping-pong effect" between actuaries and IT and address inefficiencies caused by disparate tools and languages.

**2. Solution**  
   - **Features**: Photon is a Python-based framework with a node-based graph structure that standardizes tariff modeling, integrates ML models, and enables dynamic pricing. It provides a pricing engine, graph configuration tools, and automated workflows for real-time or batch processing.
   - **Integrations**: Photon integrates with backend systems for deployment and interfaces with APIs to pull production data and support inferencing.
   - **Constraints**: Photon must operate within existing infrastructure constraints, such as AXA's microservices architecture, ensuring compatibility across business units.
   - **Out-of-Scope**: External data collection or real-time data sources unrelated to pricing adjustments are outside Photon’s scope.

**3. Data**  
   - **Sources**:
     - **Training Data**: Historical insurance data, including customer demographics, policy details, claims history, and time-based risk factors.
     - **Production Data**: Live data from customer profiles and policy updates.
   - **Labeling**: Actuarial teams prepare labeled datasets, defining premiums and risks based on existing standards, adjusted for ML-based enhancements. The data may also require preprocessing for feature engineering.

**4. Modeling**  
   - **Iterative Approach**:
     - Start with baseline actuarial models, gradually incorporate ML models (e.g., XGBoost or CatBoost) for feature prediction.
     - Use iterative testing to refine parameters, enhance accuracy, and reduce bias in pricing predictions.
     - Deploy models in a modular format within the Photon node structure to enable easy updates.

**5. Feedback**  
   - **Sources**:
     - Feedback from actuaries and business units for model adjustments.
     - Production feedback on pricing accuracy and fairness.
     - Customer response and satisfaction data on policy adjustments.

**6. Value Proposition**  
   - **Product Value**: Photon improves the speed, accuracy, and customization of insurance pricing, offering a unified platform where actuarial and IT teams work seamlessly.
   - **Pain Alleviation**: Reduces dependency on IT for pricing updates, shortens the time to market for new pricing models, and supports personalized pricing for customers, potentially improving satisfaction.

**7. Metrics**  
   - **Success Metrics**:
     - Model accuracy (e.g., RMSE for premium prediction).
     - Model deployment speed (time to integrate changes).
     - Customer satisfaction with pricing adjustments.
     - Production inference latency and reliability.

**8. Inference**  
   - **Type**: Photon supports both **real-time** (for on-demand pricing) and **batch** (for policy renewals and updates) inference.

**9. Project**  
   - **Team**: Actuarial experts, ML engineers, DevOps engineers, and IT architects.
   - **Deliverables**: Configured Photon framework, trained ML models, integration with backend, and production deployment setup.
   - **Timeline**: Multi-phase deployment over 6-12 months, starting with simpler actuarial models, followed by ML-enhanced models, and concluding with full backend integration.

**10. Feasibility**  
   - **Project Feasibility**: High feasibility given existing actuarial data and well-defined business objectives. Requires coordination among actuarial teams and IT for successful deployment.
   - **Resources Needed**: Access to training data, computational resources for model training, skilled ML and DevOps teams, and deployment support on AXA’s backend.

**11. Objectives**  
   - **Key Objectives**:
     - Develop and deploy a flexible, efficient pricing model structure.
     - Reduce tariff deployment time by 50%.
     - Enable dynamic pricing adjustments with minimal IT intervention.
     - Establish real-time and batch inferencing capabilities.

**12. Evaluation**  
   - **Offline Evaluation**:
     - Model testing against historical data to validate premium accuracy.
     - Cross-validation and A/B testing of ML models to optimize pricing predictions.
   - **Online Evaluation**:
     - Monitor production metrics such as inference latency, model drift, and customer response.
     - Continuously compare predicted premiums with real-world claims data to ensure fairness and accuracy.

---

This ML Canvas provides a clear and structured roadmap for Photon’s implementation, focusing on technical needs, business impact, and metrics for success. Let me know if you'd like to expand on any of these elements further!
