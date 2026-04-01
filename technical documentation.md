# Technical Documentation: Hybrid GNN Fraud Intelligence Journey

## 1. Goal and Thesis
Build a real-time fraud detection pipeline for mobile money ecosystems that combines graph neural networks (GNNs) with traditional tabular classification. The system aims to detect complex fraud topologies including fraud rings, synthetic identities, mule accounts, fast cash-outs, and loan fraud patterns that traditional tabular methods miss.

**Core Hypothesis:** A hybrid GNN + XGBoost model outperforms pure tabular baselines on graph-based fraud detection while maintaining operational feasibility for real-time deployment.

## 2. Repository Layout (Current Implementation Status)
- **`streaming/`**: Kafka-based transaction streaming (producer/consumer scripts - currently placeholder)
- **`ml_pipeline/data_gen/`**: Synthetic data generation with 5 fraud typologies
- **`ml_pipeline/models/`**:
  - `baseline_xgboost.py`: Tabular-only baseline using engineered features
  - `evaluate_gnn.py`: Heterogeneous GraphSAGE GNN for edge-level fraud classification
  - `stacked_hybrid.py`: Hybrid model stacking GNN probabilities with tabular features
- **`backend/`**: FastAPI skeleton with Neo4j integration (requirements: neo4j, pandas, numpy)
- **`frontend/`**: React + Tailwind CSS dashboard structure (currently placeholder)
- **`data/processed/`**: Expected location for model artifacts (final_model_data.csv, hetero_graph.pt, gnn_probabilities.csv)
- **`tests/`**: Unit tests for GNN architecture validation
- **`docs/`**: System design document with fraud typology specifications
- **`notebooks/`**: Jupyter exploration notebooks (currently placeholder)

## 3. Data Generation Pipeline (`ml_pipeline/data_gen/generate_data.py`)
**Dataset Specifications:**
- 10,000 users, 100,000 transactions, 400 agents, 5,000 devices
- 2.5% fraud rate (2,500 fraudulent transactions)
- 45-day temporal window for burst detection
- 5 distinct fraud typologies with mathematical proportions:

**Fraud Typologies Implemented:**
1. **Fraud Rings (25%)**: Cyclic transactions between 4+ users with elevated amounts
2. **Mule/SIM Swap (20%)**: Star topology with shared device identifiers
3. **Fast Cash-out (20%)**: High-velocity bursts within 60-second windows
4. **Loan Fraud (15%)**: Dense communities with homophilous default patterns
5. **Business Fraud (20%)**: Unusual densification between users and business tills

**Output:** `data/raw/p2p_transfers.csv` with columns: sender_id, receiver_id, amount, timestamp, agent_id, device_id, is_fraud, fraud_scenario

## 4. Model Scripts Analysis

### `baseline_xgboost.py` (Tabular Baseline)
**Purpose:** Establishes performance floor without graph intelligence
- **Features Used:** amount, num_accounts_linked, shared_device_flag, avg_transaction_amount, transaction_frequency, num_unique_recipients, transactions_last_24hr, round_amount_flag, night_activity_flag
- **Graph Features Excluded:** triad_closure_score, pagerank_score, in_degree, out_degree, cycle_indicator
- **Training:** XGBoost with scale_pos_weight for class imbalance (pos_weight = neg/pos)
- **Evaluation:** Scenario-specific recall analysis showing blind spots on graph-based fraud
- **Expected Weakness:** Poor detection of fraud rings and connected topologies

### `evaluate_gnn.py` (Graph-Stage Model)
**Architecture:**
- **GNNEncoder:** 2-layer GraphSAGE (64 hidden → 64 output dimensions)
- **EdgeClassifier:** Concatenates sender/receiver embeddings → 2-layer MLP for edge fraud prediction
- **HybridGNN:** Combines encoder + classifier with heterogeneous conversion (`to_hetero`)

**Training Details:**
- **Data:** PyTorch Geometric HeteroData with user nodes (13 features) and p2p edges
- **Split:** 80/20 random edge split (seed=42) maintaining temporal integrity
- **Loss:** BCEWithLogitsLoss with pos_weight = neg/pos (≈34.0 for 2.5% fraud rate)
- **Optimization:** Adam (lr=0.01), 100 epochs
- **Evaluation:** ROC-AUC, classification report, scenario-specific recall

**Key Innovation:** Heterogeneous graph handling for multi-entity fraud detection

### `stacked_hybrid.py` (Production-Ready Hybrid)
**Stacking Mechanism:**
- Concatenates tabular features with GNN edge probabilities as additional feature
- Creates hybrid feature space: [tabular_features + gnn_probability]

**Hyperparameter Tuning:**
- n_estimators=150, max_depth=4, learning_rate=0.05, colsample_bytree=0.6
- scale_pos_weight = pos_weight * 1.5 (aggressive fraud upweighting)

**Business Logic Integration:**
- **Traffic Light System:** Probability-based decision rules
  - ≥0.85: AUTO_FREEZE (instant blocking)
  - ≥0.25: MANUAL_REVIEW (analyst queue)
  - <0.25: SAFE (no action)
- **Tier-2 Handoff:** Exports `review_queue.csv` for human-AI collaboration
- **Workload Analysis:** Quantifies analyst burden and system recall

## 5. Testing Framework (`tests/test_gnn.py`)
**Unit Tests Cover:**
1. **Forward Pass Validation:** Micro-graph (3 users) inference testing
2. **Loss Computation:** BCEWithLogitsLoss with pos_weight verification
3. **Embedding Dimensions:** Validates 64D user embeddings from training
4. **Data Integrity:** Confirms 10,000 users processed correctly

**Testing Approach:** Isolated mathematical core testing without full pipeline dependencies

## 6. Technology Stack (Implemented Components)
- **ML Framework:** PyTorch Geometric, Scikit-Learn, XGBoost
- **Data Processing:** Pandas, NumPy
- **Graph Database:** Neo4j (backend integration ready)
- **API Framework:** FastAPI (skeleton implemented)
- **Frontend:** React + Tailwind CSS (structure defined)
- **Streaming:** Kafka (producer/consumer placeholders)
- **Deployment:** Docker Compose (configuration pending)

## 7. End-to-End Pipeline Flow
1. **Data Generation:** `generate_data.py` → `data/raw/p2p_transfers.csv`
2. **Feature Engineering:** Graph construction and tabular feature extraction (pending implementation)
3. **Model Training:**
   - `baseline_xgboost.py` → Tabular baseline performance
   - `evaluate_gnn.py` → GNN training → `hetero_graph.pt`, `gnn_probabilities.csv`
   - `stacked_hybrid.py` → Hybrid stacking → `review_queue.csv`
4. **Validation:** `test_gnn.py` → Architecture verification
5. **Deployment:** FastAPI serving with Neo4j persistence (pending)

## 8. Key Research Contributions
- **Empirical Proof:** Quantified performance gains on graph fraud vs tabular baselines
- **Scenario Analysis:** Per-topology recall metrics for fraud rings, fast cash-outs, etc.
- **Operational Feasibility:** Human-in-the-loop design with workload quantification
- **Scalability:** PyTorch Geometric for large graph processing
- **Explainability:** GNN architecture supports future GNNExplainer integration

## 9. Current Development Status
- ✅ Synthetic data generation with realistic fraud patterns
- ✅ Core ML models (baseline, GNN, hybrid) fully implemented
- ✅ Unit testing framework established
- ✅ Backend skeleton with Neo4j integration
- ⏳ Feature engineering pipeline (graph construction, tabular features)
- ⏳ Kafka streaming implementation
- ⏳ Frontend dashboard development
- ⏳ Docker orchestration configuration

## 10. How to Run (Current State)
```bash
# Activate virtual environment
& venv\Scripts\Activate.ps1

# Generate synthetic data
python ml_pipeline/data_gen/generate_data.py

# Run model evaluations
python ml_pipeline/models/baseline_xgboost.py
python ml_pipeline/models/evaluate_gnn.py
python ml_pipeline/models/stacked_hybrid.py

# Run tests
pytest tests/test_gnn.py
```

## 11. Future Development Roadmap
- Complete feature engineering pipeline (`ml_pipeline/features/`, `ml_pipeline/graph_builder/`)
- Implement Kafka streaming in `streaming/` folder
- Build FastAPI endpoints in `backend/`
- Develop React dashboard in `frontend/`
- Configure Docker Compose for full-stack deployment
- Add model monitoring and A/B testing capabilities

---
*Documentation auto-generated from codebase analysis. Last updated: April 1, 2026**