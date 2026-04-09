import unittest
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

print(" Running Week 5 Unit Tests ")

class TestHybridPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        This runs once before the tests start. We create dummy 'Mock' data 
        to simulate what the Kenyan mobile money pipeline looks like.
        """
        # 1. Mock Tabular Data (Like amounts and velocities)
        cls.mock_tabular = pd.DataFrame({
            'amount': [50, 1500, 120000, 300, 5000],
            'transactions_last_24hr': [1, 2, 10, 1, 3],
            'is_fraud': [0, 0, 1, 0, 1]
        })
        
        # 2. Mock GNN Probabilities (The distilled embeddings)
        cls.mock_gnn = pd.DataFrame({
            'gnn_fraud_risk_score': [0.1, 0.3, 0.95, 0.2, 0.88]
        })

    def test_1_embedding_fusion(self):
        """
        🧪 Test 1: Test Embedding Fusion
        Proves that the GNN data and Tabular data merge perfectly without losing data.
        """
        # Perform the fusion (Probability Stacking)
        fused_df = pd.concat([self.mock_tabular, self.mock_gnn], axis=1)
        
        # ASSERTIONS (The Mathematical Proofs)
        self.assertEqual(len(fused_df), 5, "Fusion failed: Row count mismatched!")
        self.assertIn('gnn_fraud_risk_score', fused_df.columns, "Fusion failed: GNN feature missing!")
        self.assertIn('amount', fused_df.columns, "Fusion failed: Tabular feature missing!")
        print("✅ Test 1 Passed: Embedding Fusion is stable.")

    def test_2_classifier_pipeline(self):
        """
        🧪 Test 2: Test Classifier Pipeline
        Proves the Hybrid XGBoost model can train and predict on the fused data.
        """
        fused_df = pd.concat([self.mock_tabular, self.mock_gnn], axis=1)
        X = fused_df.drop(columns=['is_fraud'])
        y = fused_df['is_fraud']
        
        # Initialize a small test version of your tuned Hybrid
        model = XGBClassifier(n_estimators=5, max_depth=3, random_state=42)
        
        try:
            model.fit(X, y)
            predictions = model.predict(X)
            pipeline_success = True
        except Exception as e:
            pipeline_success = False
            
        self.assertTrue(pipeline_success, "Pipeline failed: Model crashed during fit/predict!")
        self.assertEqual(len(predictions), 5, "Pipeline failed: Prediction count mismatch!")
        print("✅ Test 2 Passed: Classifier Pipeline executes without errors.")

    def test_3_risk_scoring_output_range(self):
        """
        🧪 Test 3: Test Risk Scoring Output Range
        Proves that the final probability scores are strictly between 0.0 and 1.0.
        """
        fused_df = pd.concat([self.mock_tabular, self.mock_gnn], axis=1)
        X = fused_df.drop(columns=['is_fraud'])
        y = fused_df['is_fraud']
        
        model = XGBClassifier(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X, y)
        
        # Generate the risk scores
        risk_scores = model.predict_proba(X)[:, 1]
        
        # Check the mathematical bounds
        min_score = np.min(risk_scores)
        max_score = np.max(risk_scores)
        
        self.assertGreaterEqual(min_score, 0.0, "FATAL: Risk score dropped below 0.0!")
        self.assertLessEqual(max_score, 1.0, "FATAL: Risk score exceeded 1.0!")
        print(f"✅ Test 3 Passed: Risk scores are mathematically valid (Range: {min_score:.4f} to {max_score:.4f}).")

if __name__ == '__main__':
    # Run the tests quietly, we just want to see our custom print statements
    unittest.main(verbosity=0)