

Model Nodes in Photon

Photon’s model nodes support a wide range of ML libraries, including XGBoost, CatBoost, Scikit-Learn, and specific custom nodes like Carrots Node. By implementing a GeneralMLNode, Photon can dynamically load and apply any ML model, allowing seamless integration across different pricing models without the need for library-specific nodes.

Each type of node is configured to suit particular types of data or prediction tasks, making Photon adaptable to the diverse needs of insurance pricing and customer segmentation.


---

1. GeneralMLNode: The Flexible Model Node

The GeneralMLNode is a versatile node class that can load and apply models from various libraries (such as XGBoost, CatBoost, and Scikit-Learn). This generalized approach simplifies model integration, allowing teams to specify the model name and apply it without modifying the core structure of Photon’s nodes.

class GeneralMLNode(Node):
    def compute_score(self, features, model_name):
        model = load_model(model_name)  # Load the specified ML model
        score = model.predict(features)
        return score

Example Usage: The GeneralMLNode can load a specified model for calculating scores or predictions. For instance, in a scenario where you need a claims probability or personalized score, you can load the model dynamically.

node = GeneralMLNode()
features = {"age": 30, "claims_history": 2, "vehicle_type": "SUV"}
risk_score = node.compute_score(features, model_name="xgboost_claim_model")
print(f"Risk score: {risk_score}")

Unit Test Example:

def test_general_ml_node():
    node = GeneralMLNode()
    features = {"age": 45, "claims_history": 1, "vehicle_type": "Sedan"}
    score = node.compute_score(features, model_name="xgboost_claim_model")
    assert score is not None, "Model should return a valid score."


---

2. XGBoost Node

Purpose: XGBoost is ideal for structured data and high-precision predictions, often used for tasks like claims probability prediction. Photon leverages XGBoost’s accuracy and efficiency to calculate risk scores for claims.

from xgboost import XGBClassifier

class XGBoostNode(Node):
    def predict_claim_probability(self, features):
        model = load_model("xgboost_claim_model")  # Load the XGBoost model
        probability = model.predict_proba(features)[:, 1]  # Probability of a claim
        return probability

Example Usage: The XGBoostNode might calculate the probability of a claim for a customer based on structured data, such as age and driving history. If the probability is high, Photon could adjust the premium accordingly.

node = XGBoostNode()
features = {"age": 25, "vehicle_value": 15000, "region": "Urban"}
claim_probability = node.predict_claim_probability(features)
print(f"Claim probability: {claim_probability}")

Unit Test Example:

def test_xgboost_claim_prediction():
    node = XGBoostNode()
    features = {"age": 22, "vehicle_value": 20000, "region": "High Risk"}
    probability = node.predict_claim_probability(features)
    assert 0 <= probability <= 1, "Probability should be between 0 and 1."


---

3. CatBoost Node

Purpose: CatBoost is well-suited for handling categorical data and is typically used in Photon for personalized pricing based on factors like policy type or geographic region.

from catboost import CatBoostClassifier

class CatBoostNode(Node):
    def calculate_personalized_score(self, features):
        model = load_model("catboost_personalized_model")  # Load the CatBoost model
        score = model.predict(features)
        return score

Example Usage: The CatBoostNode can adjust pricing based on categorical features like occupation and location. For instance, customers in low-risk areas might receive a premium discount.

node = CatBoostNode()
features = {"occupation": "Engineer", "location": "Rural", "policy_type": "Comprehensive"}
personalized_score = node.calculate_personalized_score(features)
print(f"Personalized premium score: {personalized_score}")

Unit Test Example:

def test_catboost_personalized_pricing():
    node = CatBoostNode()
    features = {"occupation": "Teacher", "location": "Urban", "policy_type": "Full Coverage"}
    score = node.calculate_personalized_score(features)
    assert isinstance(score, float), "Score should be a numeric value."


---

4. Scikit-Learn Node

Purpose: Scikit-Learn is used for simpler, interpretable models like logistic regression for binary classifications, such as high-risk versus low-risk customers.

from sklearn.linear_model import LogisticRegression

class ScikitLearnNode(Node):
    def predict_risk_level(self, features):
        model = load_model("sklearn_risk_model")  # Load the Scikit-Learn model
        risk_level = model.predict(features)
        return risk_level

Example Usage: A Scikit-Learn Node might classify a customer as high-risk or low-risk based on specific features. This classification could lead to premium adjustments.

node = ScikitLearnNode()
features = {"age": 30, "claims_history": 1, "vehicle_type": "Sedan"}
risk_level = node.predict_risk_level(features)
print(f"Risk level: {risk_level}")

Unit Test Example:

def test_sklearn_risk_prediction():
    node = ScikitLearnNode()
    features = {"age": 45, "claims_history": 1, "vehicle_type": "SUV"}
    risk_level = node.predict_risk_level(features)
    assert risk_level in [0, 1], "Risk level should be binary (0 or 1)."


---

5. Carrots Node

Purpose: The Carrots Node is designed to analyze customer retention by scoring customers based on demographic and behavioral data. It helps insurers identify high-risk customers and adjust premiums or provide incentives accordingly.

class CarrotsNode(Node):
    def calculate_retention_score(self, customer_data):
        model = load_model("carrots_retention_model")  # Load retention model
        score = model.predict(customer_data)
        return score

Example Usage: The CarrotsNode evaluates a retention score for customers based on features like claim history and location, which may impact retention incentives or premium adjustments.

node = CarrotsNode()
customer_data = {"age": 30, "claims_history": 2, "location": "Urban"}
retention_score = node.calculate_retention_score(customer_data)
print(f"Customer retention score: {retention_score}")

Unit Test Example:

def test_carrots_retention_score():
    node = CarrotsNode()
    customer_data = {"age": 30, "claims_history": 2, "location": "Urban"}
    score = node.calculate_retention_score(customer_data)
    assert 0 <= score <= 1, "Score should be within a probability range."


---

Summary of Model Nodes in Photon

GeneralMLNode: Flexible node structure to load any model dynamically.

XGBoost Node: Specialized for high-precision risk assessment.

CatBoost Node: Ideal for handling categorical data in personalized pricing.

Scikit-Learn Node: Useful for simple, interpretable models in binary classifications.

Carrots Node: Focused on customer retention scoring, allowing for personalized retention strategies.


These model nodes make Photon highly adaptable for a variety of ML models, supporting robust and customizable pricing models that adjust dynamically based on customer data. The setup is modular, enabling Photon to incorporate new models or update existing ones with minimal changes.


