# Technical Documentation: Hybrid GNN Fraud Intelligence Journey

## 1. Goal and Thesis
Build a real-time fraud detection pipeline for mobile money ecosystems that combines graph neural networks (GNNs) with traditional tabular classification. The system aims to detect complex fraud topologies including fraud rings, synthetic identities, mule accounts, fast cash-outs, and loan fraud patterns that traditional tabular methods miss.

**Core Hypothesis:** A hybrid GNN + XGBoost model outperforms pure tabular baselines on graph-based fraud detection while maintaining operational feasibility for real-time deployment.

## 2. Repository Layout (Current Implementation Status)
- **`streaming/`**: ✅ **Kafka-based transaction streaming**
  - `transaction_producer.py`: Synthetic M-Pesa transaction publisher to Kafka topics
  - `graph_consumer.py`: Real-time transaction consumer with Neo4j + FastAPI integration
- **`docker-compose.yml`**: ✅ **Kafka infrastructure orchestration** (Zookeeper + Kafka Broker)
- **`ml_pipeline/data_gen/`**: Synthetic data generation with 5 fraud typologies
- **`ml_pipeline/models/`**:
  - `baseline_xgboost.py`: Tabular-only baseline using engineered features
  - `evaluate_gnn.py`: Heterogeneous GraphSAGE GNN for edge-level fraud classification
  - `stacked_hybrid.py`: Hybrid model stacking GNN probabilities with tabular features
- **`backend/`**: ✅ **FastAPI production server with Neo4j & SQLite integration**
  - `main.py`: Complete API with real-time fraud detection endpoints
  - `fraud_intel.db`: SQLite database for transaction persistence
  - Dependencies: neo4j, pandas, numpy, fastapi, uvicorn
- **`frontend/`**: ✅ **React + Tailwind CSS operational dashboard**
  - All 6 pages implemented (Home, Transactions, FraudNetwork, Alerts, Reports, Settings)
  - Real-time dashboard with SQLite metrics
  - Transaction prediction form with backend integration
  - Live fraud network visualization from Neo4j
  - Alert queue management system
- **`data/processed/`**: Model artifacts (final_model_data.csv, hetero_graph.pt, gnn_probabilities.csv)
- **`models/saved/`**: Saved model weights (hybrid_xgboost.pkl - production model)
- **`tests/`**: Unit tests for GNN architecture validation
- **`docs/`**: System design document with fraud typology specifications
- **`notebooks/`**: Jupyter exploration notebooks (currently placeholder)
- **`populate_neo4j.py`**: ✅ **Batch data loader for Neo4j graph database**

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

### **Evolution of Model Development**

The project demonstrates a systematic progression from basic tabular classification to advanced hybrid graph-neural approaches, with each script representing a key milestone in the research journey.

#### **4.1 Initial XGBoost Classifier (`xgboost_classifier.py`)**
**Purpose:** Proof-of-concept tabular baseline using Neo4j data extraction
- **Data Source:** Direct Neo4j Cypher queries extracting transaction metadata
- **Features:** Transaction type, sender age, KYC level, default status
- **Architecture:** Basic XGBClassifier with manual encoding and imbalance handling
- **Evaluation:** Scenario-specific recall analysis showing limitations on graph-based fraud
- **Key Innovation:** First empirical proof that tabular methods struggle with connected fraud patterns

#### **4.2 GNN Embeddings Training (`gnn_embeddings.py`)**
**Purpose:** Generate structural embeddings from transaction graphs
- **Architecture:** 
  - `GNNEncoder`: 2-layer GraphSAGE (64D hidden → 64D output)
  - Heterogeneous conversion using `to_hetero()` for multi-entity graphs
- **Training:** Full graph training on 100K transactions with imbalance weighting
- **Output:** `user_embeddings.csv` with 64-dimensional structural features per user
- **Key Innovation:** Converts graph topology into tabular features for downstream ML

#### **4.3 Graph Dataset Construction (`graph_dataset.py`)**
**Purpose:** Convert CSV data to PyTorch Geometric tensors
- **Node Features:** 13 engineered features (tabular + graph metrics)
- **Edge Construction:** User-to-user P2P transaction edges
- **Normalization:** StandardScaler for neural network compatibility
- **Output:** `hetero_graph.pt` PyTorch HeteroData object
- **Key Innovation:** Bridges feature engineering with GNN training pipeline

#### **4.4 GNN Evaluation (`evaluate_gnn.py`)**
**Purpose:** Standalone GNN performance assessment
- **Architecture:** Same as embeddings script but with edge-level classification
- **Split:** 80/20 edge-based train/test with seed=42 reproducibility
- **Training:** 100 epochs with BCEWithLogitsLoss and pos_weight balancing
- **Evaluation:** ROC-AUC, classification report, scenario-specific recall
- **Key Innovation:** Quantifies GNN's ability to detect fraud rings missed by tabular methods

#### **4.5 Hybrid XGBoost (`hybrid_xgboost.py`)**
**Purpose:** Direct fusion of tabular features with GNN embeddings
- **Fusion Method:** Merge GNN embeddings (64D) with tabular features per sender
- **Architecture:** XGBClassifier trained on concatenated feature space
- **Evaluation:** Scenario-specific recall comparison with baseline
- **Key Innovation:** First hybrid approach proving graph features improve detection

#### **4.6 GNN Probability Extraction (`extract_gnn_probs.py`)**
**Purpose:** Generate edge-level fraud probabilities for stacking
- **Process:** Train GNN → predict on all edges → extract sigmoid probabilities
- **Output:** `gnn_probabilities.csv` with single fraud risk score per transaction
- **Key Innovation:** Distills GNN decisions into scalar features for meta-learning

#### **4.7 Stacked Hybrid (`stacked_hybrid.py`)**
**Purpose:** Production-ready stacked ensemble with business logic
- **Stacking:** Concatenate tabular features with GNN probabilities
- **Hyperparameters:** Tuned for production (150 estimators, depth=4, lr=0.05)
- **Business Logic:** Traffic light system (auto-freeze ≥0.85, review 0.25-0.85, safe <0.25)
- **Tier-2 Handoff:** Exports `review_queue.csv` for human-AI collaboration
- **Key Innovation:** Operational deployment with analyst workload optimization

#### **4.8 AI Fraud Analyst (`ai_fraud_analyst.py`)**
**Purpose:** Automated Tier-2 analysis using Kenyan behavioral rules
- **Rules Engine:** Domain-specific logic for M-Pesa fraud patterns
- **Processing:** Analyzes review queue with contextual business rules
- **Decisions:** CONFIRMED_FRAUD, AUTO_CLEARED_SAFE, REQUIRE_HUMAN
- **Impact Analysis:** Quantifies false alarm reduction and analyst workload
- **Key Innovation:** Domain expertise automation reducing human intervention

#### **4.9 Manual Inspection (`manual_inspect.py`)**
**Purpose:** Architecture validation and debugging
- **Tests:** Forward pass verification, embedding dimensions, inference on samples
- **Output:** Mathematical health checks and tensor shape validation
- **Key Innovation:** Quality assurance for GNN pipeline reliability

#### **4.10 Feature Importance Visualization (`visualize_importance.py`)**
**Purpose:** Explain hybrid model decisions
- **Method:** XGBoost feature importance on stacked feature space
- **Visualization:** Bar chart of top 10 features by F-score
- **Output:** `feature_importance.png` showing GNN probability dominance
- **Key Innovation:** Interpretability for graph-enhanced tabular models

### **Legacy Baseline XGBoost (`baseline_xgboost.py`)**
**Purpose:** Current production baseline excluding graph features
- **Features Used:** amount, num_accounts_linked, shared_device_flag, avg_transaction_amount, transaction_frequency, num_unique_recipients, transactions_last_24hr, round_amount_flag, night_activity_flag
- **Graph Features Excluded:** triad_closure_score, pagerank_score, in_degree, out_degree, cycle_indicator
- **Training:** XGBoost with scale_pos_weight for class imbalance (pos_weight = neg/pos)
- **Evaluation:** Scenario-specific recall analysis showing blind spots on graph-based fraud
- **Expected Weakness:** Poor detection of fraud rings and connected topologies

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
- **Graph Database:** Neo4j (fully integrated with live graph updates)
- **API Framework:** FastAPI with CORS middleware (production-ready)
- **Frontend:** React + Tailwind CSS with Recharts for visualizations
- **Local Persistence:** SQLite (fraud_intel.db for transaction logging)
- **Streaming:** Kafka (producer/consumer placeholders pending)
- **Deployment:** Docker Compose (configuration pending)

## 6.5 Backend API Implementation (`backend/main.py`)
**Status:** ✅ **FULLY OPERATIONAL**

### Core Infrastructure
- **Framework:** FastAPI with CORS middleware for frontend cross-origin requests
- **Database Layer:** 
  - **Neo4j:** Real-time transaction graph with live network topology updates
  - **SQLite:** Persistent transaction ledger (fraud_intel.db) for dashboard analytics and audit trails
- **Model Integration:** Loads hybrid_xgboost.pkl on startup for instantaneous prediction

### API Endpoints

#### **1. POST `/predict`**
**Purpose:** Core fraud detection engine - Tier 1 ML + Tier 2 AI Analyst

**Workflow:**
1. Receives `TransactionRequest` with tabular features (sender, receiver, amount, velocity)
2. Updates Neo4j graph with new transaction edge
3. Queries Neo4j for sender's out-degree (num_unique_recipients)
4. Constructs 16-feature vector for XGBoost hybrid model
5. Runs model inference and applies AI Analyst business logic (see Tier 2 rules)
6. Logs decision to SQLite with timestamp, reason, and risk score
7. Returns `PredictionResponse` with risk_score, decision, and Kenyan-specific reason

**Key Feature:** Integrates live Neo4j topology updates with tabular XGBoost inference for hybrid detection

#### **2. GET `/dashboard-stats`**
**Purpose:** Real-time KPI dashboard powered by SQLite aggregations

**Returns:**
- `kpis`: Total transactions, fraud count, fraud rate percentage
- `pie`: Risk distribution (Low/Medium/High) from transaction decisions
- `alerts`: Recent 4 high-risk transactions with sender, receiver, amount, status

**SQL Logic:** Excludes resolved items; counts CONFIRMED_FRAUD, AUTO_FREEZE, REQUIRE_HUMAN as active alerts

#### **3. POST `/resolve-alert/{tx_id}`**
**Purpose:** Tier 3 human analyst decision persistence

**Parameters:**
- `action`: "approve" → RESOLVED_SAFE | "deny" → RESOLVED_FRAUD

**Effect:** Updates SQLite transaction record; alerts disappear from active queue upon next fetch

#### **4. GET `/live-graph`**
**Purpose:** Fetch real transaction network from Neo4j for force-graph visualization

**Returns:** 
- Nodes: Set of all users in Neo4j (deduplicated)
- Links: SENT_MONEY edges with risk categorization based on amount (high >50K, medium >5K, low ≤5K)

**Query:** MATCH (s:User)-[r:SENT_MONEY]->(t:User) LIMIT 50 for scalability

### AI Analyst Tier 2 Rules (`apply_ai_analyst` function)
Implements domain-specific M-Pesa fraud detection business logic:
- **≥0.85 risk score:** AUTO_FREEZE (high confidence severe topology)
- **0.50 < score with amount <300 & velocity >5:** CONFIRMED_FRAUD (micro-scam/Kamiti rule)
- **score <0.50, amount 100-3000, velocity <4:** AUTO_CLEARED_SAFE (kiosk retail pattern)
- **amount >100K:** REQUIRE_HUMAN (washwash compliance limit)
- **Otherwise:** REQUIRE_HUMAN (ambiguous)

### Database Schema (SQLite)
```sql
CREATE TABLE transactions (
    id INTEGER PRIMARY KEY,
    transaction_id TEXT,
    timestamp DATETIME,
    sender_id TEXT,
    receiver_id TEXT,
    amount REAL,
    risk_score REAL,
    decision TEXT,
    reason TEXT
)
```

## 6.6 Frontend Dashboard Implementation (`frontend/`)
**Status:** ✅ **FULLY OPERATIONAL**

### Technology Stack
- **Framework:** React 18 with React Router for multi-page navigation
- **UI Components:** Tailwind CSS with custom brand colors (brandPrimary: #6366f1, brandDark: #1f2937)
- **Visualizations:** Recharts for complex charts (AreaChart, PieChart)
- **Icons:** Lucide React for consistent iconography
- **HTTP Client:** Axios for FastAPI integration

### Page Implementations

#### **Home Dashboard (`pages/Home.jsx`)**
- **Auto-refresh:** Fetches SQLite stats every 5 seconds from `/dashboard-stats`
- **Widgets:**
  - KPI cards: Total transactions, fraud count, fraud rate
  - Pie chart: Risk distribution (Low/Medium/High)
  - Area chart: Hourly transaction volume (mocked for prototype clarity)
  - Recent alerts: Latest 4 flagged transactions with status badges

#### **Transactions (`pages/Transaction.jsx`)**
- **Form Interface:** Allows manual transaction submission for testing
- **Fields:** transaction_id, sender_id, receiver_id, amount, transactions_last_24hr, hour
- **Integration:** POSTs to `/predict` endpoint
- **Response Display:** Shows decision (AUTO_FREEZE, CONFIRMED_FRAUD, AUTO_CLEARED_SAFE, REQUIRE_HUMAN), risk percentage, and AI explanation

#### **Fraud Network (`pages/FraudNetwork.jsx`)**
- **Case Studies:** 3 pre-loaded fraud ring case studies with hard-coded node/link data
  - Case 1: Agent Reversal Scam Ring (directed cycle + fan-in)
  - Case 2: Mulot SIM Swap Mules (star topology)
  - Case 3: Loan Fraud Community (dense subgraph)
- **Live Mode:** Queries `/live-graph` endpoint to visualize real Neo4j transaction network
- **Visualization:** react-force-graph-2d for physics-based force-directed layout
- **Node Styling:** Nodes color-coded by risk level

#### **Alerts (`pages/Alerts.jsx`)**
- **Queue System:** Fetches pending transactions from `/dashboard-stats`
- **UI Layout:** Split view (left: queue, right: detailed case)
- **Actions:** Human analyst can approve (RESOLVED_SAFE) or deny (RESOLVED_FRAUD)
- **Endpoint:** POSTs to `/resolve-alert/{tx_id}?action={approve|deny}`
- **Inbox Zero:** Shows success state when all alerts cleared

#### **Reports (`pages/Reports.jsx`)**
- **Compliance Dashboard:** Mock CBK (Central Bank of Kenya) AML report generation
- **Cards:** Model accuracy (96.4%), drift detection, compliance status
- **Export:** Button for encrypted PDF compliance export (prototype alert)

#### **Settings (`pages/Settings.jsx`)**
- **Configuration Tabs:** Profile, Notifications, Security, Data & Privacy, Email Preferences, API Keys
- **Dynamic Thresholds:** High/medium risk level sliders
- **Toggles:** High-risk alerts, daily digests, system notifications

### Layout Component (`components/Layout.jsx`)
- **Sidebar Navigation:** Brand logo, main navigation menu with icons
- **Active Route Highlighting:** Visual feedback for current page
- **Responsive:** Content area expands; sidebar fixed on desktop

### Frontend-Backend Communication
- **Base URL:** http://127.0.0.1:8000 (configurable via axios)
- **CORS:** Enabled on backend for all origins
- **Error Handling:** Try-catch blocks with user-friendly error messages
- **Real-time Updates:** Home dashboard auto-refreshes every 5 seconds for live KPI feel

## 6.7 Data Loading & Neo4j Integration (`populate_neo4j.py`)
**Status:** ✅ **READY FOR DEPLOYMENT**

### Functionality
- **Bulk UNWIND Inserts:** Uses Cypher UNWIND for efficient batch loading
- **Auto-ID Generation:** If transaction_id missing from CSV, auto-generates TXN_0, TXN_1, etc.
- **Deduplication:** MERGE ensures no duplicate users or transaction edges
- **Database Cleanup:** Optional clear function to wipe data before fresh load

### Data Pipeline
1. Loads `data/processed/final_model_data.csv` (output from data generation)
2. Batches rows for efficient Neo4j writes
3. Creates User nodes and SENT_MONEY transaction edges with amounts
4. Enables live graph visualization and real-time topology queries

## 6.8 Kafka Streaming Implementation (`streaming/`)
**Status:** ✅ **FULLY OPERATIONAL**

### Architecture Overview
Real-time transaction streaming pipeline with Kafka as the event backbone:
1. **Kafka Broker** (port 9092) — Centralized event hub
2. **Transaction Producer** — Publishes synthetic M-Pesa transactions every 2 seconds
3. **Graph Consumer** — Subscribes to transaction stream, updates Neo4j graph, calls FastAPI `/predict` endpoint
4. **Neo4j** — Live graph accumulates transactions with SENT_MONEY edges
5. **FastAPI** — Fraud scoring triggered per transaction
6. **SQLite** — Decision logging for audit trail

### Docker Compose Infrastructure (`docker-compose.yml`)

**Services:**
- **Zookeeper** (image: confluentinc/cp-zookeeper:7.0.0)
  - Port: 2181 (internal cluster coordination)
  - Environment: ZOOKEEPER_CLIENT_PORT=2181
  - Purpose: Manages broker leader election and partition metadata

- **Kafka Broker** (image: confluentinc/cp-kafka:7.0.0)
  - Port: 9092 (external client connections)
  - Port: 29092 (internal broker communication)
  - Environment: Auto-topic creation enabled (`AUTO_CREATE_TOPICS_ENABLE=true`)
  - Broker ID: 1
  - Zookeeper dependency: localhost:2181

**Bootstrap:**
```bash
docker-compose up -d
# Verify services running: docker-compose ps
# Logs: docker-compose logs -f kafka
```

### Transaction Producer (`streaming/transaction_producer.py`)

**Purpose:** Simulate real-time M-Pesa transaction stream

**Functionality:**
- Connects to Kafka broker at `localhost:9092`
- Generates synthetic transactions every 2 seconds
- Publishes to `transactions` topic (auto-created by Kafka)
- Transaction schema:
  ```
  {
    "transaction_id": "TXN_XXXXXX",
    "timestamp": "2026-04-09T15:30:45.000Z",
    "sender_id": "USER_1234",
    "receiver_id": "USER_5678",
    "amount": 15000.00,
    "device_id": "DEVICE_ABC",
    "agent_id": "AGENT_001"
  }
  ```

**Execution:**
```bash
python streaming/transaction_producer.py
# Output: "Sent transaction TXN_XXXXXX: sender_1234 → receiver_5678 (15000 KES)"
```

### Graph Consumer (`streaming/graph_consumer.py`)

**Purpose:** Real-time integration of Kafka transactions with Neo4j and fraud detection

**Dual Update Workflow:**

1. **Neo4j Graph Update:**
   - Consumes transaction from Kafka topic `transactions`
   - Creates/updates User nodes if not exists (sender, receiver)
   - Creates SENT_MONEY edge from sender → receiver with amount metadata
   - Graph grows organically as transactions stream in
   - Enables live network visualization via `/live-graph` endpoint

2. **FastAPI Prediction Call:**
   - Extracts transaction features from Kafka message
   - Constructs feature vector compatible with hybrid_xgboost.pkl
   - POSTs to `http://127.0.0.1:8000/predict` endpoint
   - Receives risk_score, decision (AUTO_FREEZE, CONFIRMED_FRAUD, etc.), and reason
   - Logs result to console with timestamp

**Consumer Configuration:**
- Kafka broker: `localhost:9092`
- Topic: `transactions`
- Group: `fraud-detection-group` (enables consumer offset tracking)
- Auto-commit: Enabled (commits offset after successful prediction)

**Execution:**
```bash
python streaming/graph_consumer.py
# Output example:
# Consumer started, listening to transactions topic...
# [15:30:45] Received TXN_XXXXXX from sender_1234
# Neo4j updated with SENT_MONEY edge
# [15:30:46] FastAPI /predict called, risk_score=0.72, decision=REQUIRE_HUMAN
```

### Data Flow Integration

```
[Transaction Producer]
         ↓
   Kafka Topic: transactions
         ↓
[Graph Consumer]
    ├→ Neo4j (graph updated)
    └→ FastAPI /predict
         ├→ Hybrid XGBoost inference
         ├→ AI Analyst Tier 2 rules
         └→ SQLite logging
              ↓
   [Dashboard refresh]
```

### Key Design Decisions

| Aspect | Choice | Rationale |
|--------|--------|----------|
| **Message Format** | JSON strings | Human-readable, easy to debug, standard Kafka convention |
| **Topic Name** | `transactions` | Semantic; matches M-Pesa use case |
| **Consumer Group** | `fraud-detection-group` | Enables horizontal scaling; multiple consumers can process different partitions |
| **Offset Commit** | Auto-commit enabled | Simplifies development; production should use manual commits for fine-grained control |
| **Publish Frequency** | 2 seconds | Balances realistic streaming with manageable volume for testing |
| **Error Handling** | Log & continue | Prevents cascading failures; failed predictions don't block Kafka consumption |

### Integration with Existing Components

**Backend (`backend/main.py`):** 
- No changes needed; FastAPI listens for consumer POSTs to `/predict`
- Consumer calls: `POST http://127.0.0.1:8000/predict`
- Returns fraud decision to consumer for logging

**Frontend (`frontend/`):**
- No direct Kafka coupling; consumes via FastAPI and SQLite
- Dashboard updates via `/dashboard-stats` endpoint (SQLite aggregations)
- No latency requirement; 5-second refresh rate sufficient

**Neo4j:**
- Consumer updates graph in real-time
- `/live-graph` endpoint returns latest topology for FraudNetwork page
- Enables dynamic fraud ring visualization as transactions stream

## 7. End-to-End Pipeline Flow
1. **Data Generation:** `generate_data.py` → `data/raw/p2p_transfers.csv`
2. **Feature Engineering:** Graph construction and tabular feature extraction
3. **Model Training:**
   - `baseline_xgboost.py` → Tabular baseline performance
   - `evaluate_gnn.py` → GNN training → `hetero_graph.pt`, `gnn_probabilities.csv`
   - `stacked_hybrid.py` → Hybrid stacking → `hybrid_xgboost.pkl` (saved model)
4. **Neo4j Population:** `populate_neo4j.py` → Batch-loads transactions into graph database
5. **Docker Infrastructure:** `docker-compose up -d` → Zookeeper + Kafka Broker startup
6. **API Deployment:** `backend/main.py` (FastAPI)
   - Loads hybrid model from pickle
   - Initializes SQLite fraud_intel.db
   - Starts listening on http://127.0.0.1:8000
7. **Kafka Streaming Pipeline:**
   - `transaction_producer.py` → Publishes synthetic M-Pesa transactions every 2 seconds to Kafka topic
   - `graph_consumer.py` → Subscribes to Kafka, updates Neo4j in real-time, calls FastAPI `/predict`
8. **Frontend Deployment:** `frontend/` (React)
   - Starts Vite dev server
   - Connects to FastAPI backend
   - Displays real-time dashboard with Neo4j network visualization
9. **Live Detection Pipeline:**
   - Kafka producer generates synthetic transaction
   - Consumer receives transaction, updates Neo4j graph
   - Consumer calls FastAPI `/predict` endpoint
   - Hybrid XGBoost inference + Tier 2 AI Analyst rules applied
   - Response + decision logged to SQLite
   - Dashboard auto-refreshes with updated metrics
   - Analyst reviews alerts in queue and resolves via UI
10. **Validation:** `test_gnn.py`, `test_hybrid_pipeline.py` → Architecture verification

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
- ✅ **Backend FastAPI with full Neo4j integration (COMPLETE)**
- ✅ **Frontend React dashboard with all pages (COMPLETE)**
- ✅ **SQLite transaction database for persistence (COMPLETE)**
- ✅ **Tier 2 AI Analyst rules engine operational (COMPLETE)**
- ✅ **Live fraud network visualization from Neo4j (COMPLETE)**
- ✅ **Transaction prediction endpoint with hybrid model (COMPLETE)**
- ✅ **Alerts queue management system (COMPLETE)**
- ✅ **Dashboard analytics and KPI tracking (COMPLETE)**
- ✅ **Kafka streaming with real-time transaction producer/consumer (COMPLETE)**
- ✅ **Docker Compose for Zookeeper + Kafka infrastructure (COMPLETE)**
- ✅ **Full end-to-end streaming pipeline with Neo4j + FastAPI integration (COMPLETE)**

## 9.5 Recent Implementation Highlights (April 2026)

### Backend Achievements
- **Live Graph Integration:** FastAPI seamlessly integrates with Neo4j using UNWIND batch queries
- **Hybrid Model Serving:** XGBoost model loads on server startup for zero-latency inference
- **Persistent Storage:** All predictions logged to SQLite with timestamps, decisions, and AI reasoning
- **Multi-Tier Validation:** Tier 1 (ML model) + Tier 2 (AI Analyst rules) + Tier 3 (human review) workflow
- **Neo4j Topology Updates:** Real-time sender out-degree calculation for num_unique_recipients feature
- **CORS Enabled:** Full cross-origin request support for frontend-backend communication

### Frontend Achievements
- **Real-Time Dashboard:** Auto-refreshing KPI cards with SQLite aggregations every 5 seconds
- **Interactive Prediction:** Manual transaction form with instant risk assessment display
- **Network Visualization:** Force-graph rendering of fraud rings with case studies and live Neo4j data
- **Alert Management:** Analyst queue system with granular approve/deny actions
- **Responsive Design:** Mobile-friendly layout with Tailwind grid system
- **Compliance Reports:** CBK AML report templates and audit trail UI

### Testing Validated
- ✅ FastAPI endpoints operational (POST /predict, GET /dashboard-stats, POST /resolve-alert, GET /live-graph)
- ✅ SQLite schema and CRUD operations confirmed
- ✅ Neo4j bulk loading with transaction_id auto-generation
- ✅ React routing and component state management
- ✅ Hybrid model pickle serialization and deserialization
- ✅ CORS middleware passing browser pre-flight checks

## 10. How to Run (Current State)

### Step 1: Start Kafka Infrastructure
```bash
# Verify Docker running
docker-compose up -d

# Verify services running
docker-compose ps
# Output: zookeeper RUNNING, kafka RUNNING
```

### Step 2: Activate Environment & Generate Data
```bash
# Activate virtual environment
& venv\Scripts\Activate.ps1

# Generate synthetic data (creates data/raw/p2p_transfers.csv)
python ml_pipeline/data_gen/generate_data.py

# Build graph dataset tensor
python ml_pipeline/models/graph_dataset.py

# Extract GNN probabilities for hybrid model
python ml_pipeline/models/extract_gnn_probs.py

# Run full model training pipeline
python ml_pipeline/models/stacked_hybrid.py   # Produces hybrid_xgboost.pkl
python ml_pipeline/models/ai_fraud_analyst.py # Tier 2 analysis
```

### Step 3: Load Data into Neo4j
```bash
# Before running: Start Neo4j Desktop with credentials (uri: neo4j://localhost:7687, auth: neo4j/12345678)
python populate_neo4j.py   # Bulk-loads transactions into Neo4j graph
```

### Step 4: Start Backend API Server
```bash
cd backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
# API now live at http://127.0.0.1:8000
# Swagger docs: http://127.0.0.1:8000/docs
```

### Step 5: Start Kafka Streaming Pipeline (2 terminals)
```bash
# Terminal A: Transaction Producer
python streaming/transaction_producer.py
# Output: "Sent transaction TXN_XXXXXX: ..." every 2 seconds

# Terminal B: Graph Consumer
python streaming/graph_consumer.py
# Output: "Received TXN_XXXXXX from sender_1234..."
#         "Neo4j updated with SENT_MONEY edge"
#         "FastAPI /predict called, decision=..."
```

### Step 6: Start Frontend React Dashboard
```bash
cd frontend
npm install  # If first time
npm run dev  # Starts Vite dev server (default: http://localhost:5173)
```

### Step 7: Test the Full Stack
**Option A: Live Streaming Transaction Detection**
- Kafka producer auto-generates M-Pesa transactions
- Consumer updates Neo4j graph + calls FastAPI in real-time
- Navigate to http://localhost:5173/network → "LIVE" mode
- Watch fraud network grow as transactions stream
- Check Alerts queue for flagged transactions

**Option B: Use Transaction Form** (sends individual transactions)
- Navigate to http://localhost:5173/transactions
- Fill form and submit
- Observe risk score, decision, and AI reasoning
- Check Neo4j graph updated in /network page

**Option C: Use Alerts Queue**
- Transactions from Kafka producer appear in queue
- Navigate to /alerts to see pending review queue
- Approve/deny decisions persist to SQLite
- Check /reports for compliance metrics

**Option D: Monitor Kafka Pipeline**
```bash
# In separate terminal, watch Kafka messages
docker exec -it <kafka_container_id> kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic transactions \
  --from-beginning
```

### Full Stack Verification Checklist
- ✅ Zookeeper + Kafka running: `docker-compose ps`
- ✅ Neo4j graph has 100k+ nodes: Open Neo4j Desktop → query `MATCH (n) RETURN count(n)`
- ✅ FastAPI responding: `curl http://127.0.0.1:8000/docs` → Swagger UI loads
- ✅ Kafka producer generating: Check console output every 2 seconds
- ✅ Kafka consumer consuming: Check consumer logs show "Received TXN_XXXXXX..."
- ✅ Neo4j updating live: Query `MATCH (u)-[r:SENT_MONEY]->(v) RETURN count(r)` → count increasing
- ✅ SQLite persisting: Check `backend/fraud_intel.db` → transactions table growing
- ✅ Frontend displaying: http://localhost:5173 → Home dashboard shows live KPIs

### Automated Pipeline (Optional)
```bash
# Run full stack test without manual steps
python tests/test_hybrid_pipeline.py  # Validates all components
```

## 11. Critical Prerequisites & Configuration

### Environment Setup
- **Python 3.8+** with virtualenv
- **Node.js 16+** for npm/Vite frontend
- **Neo4j Desktop 5.x+** with active graph instance listening on neo4j://localhost:7687
- **Port Availability:** Ensure ports 8000 (FastAPI), 5173 (Vite), 7687 (Neo4j) are free

### Backend Configuration (backend/main.py)
```python
# Update these credentials if using remote Neo4j:
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "12345678")  # Change password if Neo4j password differs

# Model path (auto-resolves to models/saved/hybrid_xgboost.pkl)
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved", "hybrid_xgboost.pkl")
```

### Frontend Configuration
- Backend URL: http://127.0.0.1:8000 (hardcoded in axios calls; update if deployment differs)
- Vite port: 5173 (editable in vite.config.js)

### Database Schema Initialization
- **SQLite:** Auto-created on first FastAPI startup (fraud_intel.db)
- **Neo4j:** Requires manual data load via `populate_neo4j.py` after startup

## 12. Known Limitations & Future Work

### Current Limitations
- **Horizontal Scaling:** Current implementation runs on single machine; multi-broker Kafka cluster untested
- **Consumer Error Handling:** Failed FastAPI predictions logged but not retried; no dead letter queue
- **Topic Partitioning:** Single partition per topic; production should partition for parallel consumption
- **Monitoring Dashboard:** No Kafka metrics in frontend; Kafka broker metrics not surfaced to UI
- **Production Protocols:** Plain-text Kafka communication; production should use SSL/TLS
- **Scalability:** Current implementation tested on single-machine setup; multi-instance deployment untested
- **Authentication:** No authentication layer; API is open to localhost
- **Frontend Deployment:** Vite used for development; production build and CDN deployment pending
- **GNN Mock Score:** Backend uses hardcoded mock_gnn_score = 0.45; should replace with actual GNN inference
- **Historical Data:** Dashboard area chart is mocked; real time-series aggregation from SQLite TODO

### Next Steps (Priority Order)
1. **Production Vite Build:** Create optimized build for deployment
2. **Docker Containerization:** Full docker-compose with PostgreSQL, Redis, Neo4j services
3. **Kubernetes Orchestration:** StatefulSet for Neo4j, Deployment for FastAPI/React
4. **Kafka Integration:** Wire streaming producer/consumer for real-time transaction ingestion
5. **Authentication/Authorization:** JWT tokens, role-based access for analyst tiers
6. **Monitoring & Observability:** Prometheus metrics, ELK stack for logs, Grafana dashboards
7. **Batch Predictions:** Endpoint for daily bulk model inference on transaction archives
8. **A/B Testing Framework:** Compare hybrid vs baseline model performance on live data
9. **Model Retraining Pipeline:** Scheduled retraining with new fraud patterns
10. **Compliance Audit Trail:** Immutable ledger of all analyst decisions for regulatory review

# Generate feature importance visualization
python ml_pipeline/models/visualize_importance.py

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

## 12. Model Evolution Summary

The `ml_pipeline/models/` folder contains 11 scripts representing a complete research-to-production pipeline:

**Phase 1: Foundation (Neo4j Integration)**
- `xgboost_classifier.py`: Initial proof-of-concept with database queries

**Phase 2: Graph Learning**
- `graph_dataset.py`: Data preparation for GNN training
- `gnn_embeddings.py`: Structural embedding generation
- `manual_inspect.py`: Architecture validation

**Phase 3: Hybrid Approaches**
- `baseline_xgboost.py`: Tabular baseline (upgraded)
- `hybrid_xgboost.py`: Direct embedding fusion
- `extract_gnn_probs.py`: Probability distillation for stacking
- `evaluate_gnn.py`: Standalone GNN evaluation
- `stacked_hybrid.py`: Production-ready stacked ensemble

**Phase 4: Operational Intelligence**
- `ai_fraud_analyst.py`: Automated Tier-2 analysis
- `visualize_importance.py`: Model interpretability

**Key Progression:**
1. **Tabular → Graph**: From basic features to structural embeddings
2. **Fusion → Stacking**: From direct concatenation to meta-learning
3. **Research → Production**: From evaluation to operational deployment
4. **Single Model → System**: From ML to human-AI collaboration

This evolution demonstrates the systematic development of hybrid GNN-XGBoost fraud detection, from initial experiments to production deployment with business logic integration.*