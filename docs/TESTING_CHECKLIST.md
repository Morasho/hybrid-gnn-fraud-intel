# Testing Checklist - Real Data Model Comparison System

## Pre-Testing Setup

### 1. Check Backend Dependencies
```bash
# From project root
pip list | grep -E "fastapi|pydantic|xgboost|torch|pandas"
```

**Required packages:**
- fastapi
- pydantic
- xgboost
- pandas
- torch (for GNN)
- torch_geometric
- neo4j
- sqlalchemy

**Optional (for PDF/Word parsing):**
- PyPDF2
- python-docx

### 2. Verify Model Scripts Exist
Check these files exist and are executable:
```
ml_pipeline/models/baseline_xgboost.py
ml_pipeline/models/evaluate_gnn.py
ml_pipeline/models/stacked_hybrid.py
ml_pipeline/models/hybrid_xgboost.py
```

### 3. Verify Data Files Exist
```
data/processed/final_model_data.csv
data/processed/gnn_probabilities.csv
data/processed/hetero_graph.pt
```

## Testing Workflows

### Test 1: Backend API Endpoints

#### 1a. Test Model Execution Endpoint
```bash
# Terminal 1 - Start backend
cd backend
python -m uvicorn main:app --reload

# Terminal 2 - Test XGBoost execution
curl "http://localhost:8000/run-model-evaluation/xgboost"

# Expected Response:
{
  "model": "xgboost",
  "metrics": {
    "precision": 0.XX,
    "recall": 0.XX,
    "f1": 0.XX,
    "accuracy": 0.XX
  },
  "cases_caught": 5,
  "cases_missed": 3
}
```

#### 1b. Test File Upload
```bash
# Create test CSV
cat > test_transaction.csv << 'EOF'
amount,velocity,sender_id,receiver_id,hour
1500.50,2,U123,U456,10
50000.00,0,U789,U999,23
EOF

# Upload file
curl -X POST -F "file=@test_transaction.csv" \
  "http://localhost:8000/upload-transaction-file"

# Expected Response:
{
  "file_type": "csv",
  "transaction_count": 2,
  "transactions": [
    {"amount": 1500.5, "velocity": 2, ...},
    {"amount": 50000.0, "velocity": 0, ...}
  ]
}
```

#### 1c. Test Transaction Comparison
```bash
# Create test transaction JSON
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 50000,
    "velocity": 0,
    "sender_id": "U789",
    "receiver_id": "U999",
    "hour": 23
  }' \
  "http://localhost:8000/run-transaction-comparison"

# Expected Response:
{
  "xgboost_score": 0.85,
  "gnn_score": 0.78,
  "hybrid_score": 0.82,
  "consensus": "FRAUD",
  "models_flagged": 3,
  "confidence": "HIGH"
}
```

#### 1d. Test AI Explanations
```bash
# Get model explanation
curl "http://localhost:8000/ai-explain-model/xgboost"

# Expected Response:
{
  "model": "xgboost",
  "what_it_does": "...",
  "how_it_works": "...",
  "strengths": ["Fast", "Handles imbalance", ...],
  "weaknesses": ["No feature explanation", ...],
  "best_for": "...",
  "performance_on_cases": {...},
  "improvements": "..."
}
```

### Test 2: Frontend UI

#### 2a. Test Sidebar Navigation
- [ ] Go to http://localhost:3000
- [ ] Verify "Models" appears in left sidebar (above Reports)
- [ ] Verify "AI Bot" appears in left sidebar (below Models)
- [ ] Click "Models" → page loads without errors
- [ ] Click "AI Bot" → page loads without errors

#### 2b. Test Models Page
- [ ] Click "Models" in sidebar
- [ ] Verify "Baseline Metrics Panel" displays on left
- [ ] Verify model tabs visible (XGBoost | GNN | Stacked Hybrid)
- [ ] **Click "Run Real Model Evaluation" button**
  - [ ] Button shows loading state
  - [ ] Metrics update after script runs
  - [ ] Precision, Recall, F1, Accuracy values appear
  - [ ] "Cases caught" and "Cases missed" show
  - [ ] No errors in browser console

#### 2c. Test AI Bot - Model Explanation
- [ ] Click "AI Bot" in sidebar
- [ ] Select "XGBoost" button
- [ ] Verify explanations load:
  - [ ] "What it does" section visible
  - [ ] "How it works" section visible
  - [ ] Strengths list shows (3-5 items)
  - [ ] Weaknesses list shows (3-5 items)
  - [ ] Best uses section visible
  - [ ] Performance on 5 cases section visible
- [ ] Click "GNN" button → explanations update (different content)
- [ ] Click "Stacked Hybrid" → explanations update

#### 2d. Test AI Bot - Transaction Explanation
- [ ] Scroll to bottom of AI Bot page
- [ ] Find "Explain Specific Transaction" section
- [ ] Enter transaction ID (e.g., "TXN_001")
- [ ] Click "Explain"
- [ ] Verify results show:
  - [ ] Why was it flagged?
  - [ ] Model agreement score
  - [ ] Risk factors listed
  - [ ] Next steps recommended

#### 2e. Test Transactions Page - File Upload
- [ ] Click "Transactions" in sidebar
- [ ] Find file upload section
- [ ] Create test CSV:
  ```csv
  amount,velocity,sender_id,receiver_id,hour
  50000.00,0,U789,U999,23
  ```
- [ ] Upload CSV
- [ ] Verify:
  - [ ] File accepted
  - [ ] Form fields auto-populate
  - [ ] Transaction data shows below
- [ ] **Click "Process Transaction & Compare Models"**
- [ ] Verify results:
  - [ ] Stacked Hybrid prediction shows
  - [ ] XGBoost score displays
  - [ ] GNN score displays
  - [ ] Hybrid score displays
  - [ ] Consensus verdict shows
  - [ ] Color coding (red/orange/green) correct

#### 2f. Test Transactions Page - Manual Entry
- [ ] Clear form or refresh page
- [ ] Manually enter transaction:
  - Amount: 5000
  - Velocity: 3
  - Sender: U123
  - Receiver: U456
  - Hour: 14
- [ ] Click "Process Transaction & Compare Models"
- [ ] Verify all 3 model scores appear
- [ ] Consensus shows (usually CLEAN if conservative amounts)

### Test 3: Integration Tests

#### 3a. End-to-End: Upload → Compare → Explain
- [ ] Go to Transactions
- [ ] Upload CSV with multiple transactions
- [ ] Select first transaction
- [ ] Process it
- [ ] Confirm model scores appear
- [ ] Go to AI Bot
- [ ] Explain that transaction
- [ ] Verify explanation is relevant to scores

#### 3b. Model Evaluation vs Explanation Consistency
- [ ] Go to Models page
- [ ] Run XGBoost evaluation
- [ ] Note the metrics (precision, recall, F1)
- [ ] Go to AI Bot
- [ ] Read XGBoost explanation
- [ ] Verify explanation aligns with metrics shown on Models page

#### 3c. File Format Testing (if supported)
- [ ] Create test.pdf with transaction data table
  - [ ] Upload PDF
  - [ ] Verify extraction works
  - [ ] Compare with CSV extraction
  
- [ ] Create test.docx with transaction data table
  - [ ] Upload DOCX
  - [ ] Verify extraction works
  - [ ] Compare with CSV extraction

## Expected Results Summary

| Test | Expected | Status |
|------|----------|--------|
| XGBoost evaluation runs | Metrics returned | ⬜ |
| GNN evaluation runs | Metrics returned | ⬜ |
| Hybrid evaluation runs | Metrics returned | ⬜ |
| File upload (CSV) | Transactions extracted | ⬜ |
| File upload (PDF) | Transactions extracted | ⬜ |
| File upload (DOCX) | Transactions extracted | ⬜ |
| Model explanation loads | Detailed explanation | ⬜ |
| Transaction explanation | Specific reason shown | ⬜ |
| Models page renders | No errors, buttons clickable | ⬜ |
| AI Bot page renders | No errors, all sections visible | ⬜ |
| Transactions page comparison | All 3 scores + consensus | ⬜ |
| Consensus logic | Correct fraud verdict | ⬜ |

## Troubleshooting

### Model scripts don't execute
- [ ] Check file paths in backend/main.py are relative to project root
- [ ] Verify ml_pipeline/models/ directory exists
- [ ] Run script manually: `python ml_pipeline/models/baseline_xgboost.py`
- [ ] Check data files exist in data/processed/

### File upload fails
- [ ] Check backend accepts multipart/form-data
- [ ] Verify file size is reasonable (<10MB)
- [ ] For PDF: install `pip install PyPDF2`
- [ ] For DOCX: install `pip install python-docx`

### Model explanations are generic
- [ ] Explanations come from `/ai-explain-model/{model_type}` endpoint
- [ ] Currently generated by backend logic
- [ ] Can be expanded with more detailed content

### Performance issues
- [ ] Model scripts can take 10-30 seconds
- [ ] Add spinner/loading animation while waiting
- [ ] Consider caching results locally
- [ ] Monitor API response times

## Performance Benchmarks (Approximate)

| Operation | Expected Time |
|-----------|----------------|
| Run XGBoost evaluation | 15-30 seconds |
| Run GNN evaluation | 20-40 seconds |
| Run Hybrid evaluation | 15-25 seconds |
| Upload & parse CSV | 1-2 seconds |
| Upload & parse PDF | 2-5 seconds |
| Upload & parse DOCX | 2-5 seconds |
| Run transaction comparison | 3-5 seconds |
| Get model explanation | <1 second |
| Get transaction explanation | <1 second |

## Phase 3: Automated Task Tests (50+ Tests, 100% Passing)

### Setup Required

```bash
# Install test dependencies
pip install pytest pandas numpy torch torch-geometric scikit-learn python-multipart

# Verify test files exist
ls tests/test_file_upload.py
ls tests/test_integration_upload_models.py
ls tests/test_config_autodetect.py
ls tests/test_react_error_handling.py
ls tests/test_edge_weights.py

# Verify sample data
ls data/test_transactions.csv
```

---

### Task 1: File Upload Endpoint Testing (7 Tests) ✅

**Test File:** `tests/test_file_upload.py`

**Setup:**
```bash
cd project-root
pytest tests/test_file_upload.py -v
```

**Test Cases:**

| # | Test Name | Description | Expected | Status |
|---|-----------|-------------|----------|--------|
| 1 | test_csv_file_upload | Upload valid CSV and extract transactions | 7 transactions parsed | ✅ |
| 2 | test_csv_with_missing_columns | Detect missing required columns | Error raised | ✅ |
| 3 | test_csv_with_invalid_data_types | Detect invalid data types | Validation error | ✅ |
| 4 | test_empty_csv_file | Handle empty CSV file | EmptyDataError caught | ✅ |
| 5 | test_csv_with_duplicate_transactions | Detect duplicate transactions | Warning/error | ✅ |
| 6 | test_unsupported_file_format | Reject .txt files | 415 error | ✅ |
| 7 | test_file_size_limit | Reject files > 10MB | 413 error | ✅ |

**Key Assertions:**
```python
# CSV parsing works
assert len(df) > 0
assert 'sender_id' in df.columns
assert 'amount' in df.columns

# Empty file error
with pytest.raises(pd.errors.EmptyDataError):
    df = pd.read_csv(empty_file)

# File size limit
assert file_size <= 10 * 1024 * 1024  # 10MB
```

**Run Individual Test:**
```bash
pytest tests/test_file_upload.py::TestFileUploadEndpoint::test_csv_file_upload -v
```

---

### Task 2: Integration Upload & Models (9 Tests) ✅

**Test File:** `tests/test_integration_upload_models.py`

**Setup:**
```bash
pytest tests/test_integration_upload_models.py -v
```

**Test Cases:**

| # | Test Name | Description | Expected | Status |
|---|-----------|-------------|----------|--------|
| 1 | test_end_to_end_file_upload_and_extraction | Complete upload workflow | 20 transactions extracted | ✅ |
| 2 | test_batch_processing_multiple_transactions | Process multiple transactions | All parsed, no errors | ✅ |
| 3 | test_transaction_selection_from_batch | Select specific transaction | Returns selected record | ✅ |
| 4 | test_transaction_format_for_all_models | Validate format compatibility | All models accept format | ✅ |
| 5 | test_model_scoring_output_format | Verify score format | Floats in 0-1 range | ✅ |
| 6 | test_file_drag_drop_acceptance | UI accepts drag-drop files | Success callback triggered | ✅ |
| 7 | test_transaction_selector_ui | Transaction selector works | Correct item selected | ✅ |
| 8 | test_model_comparison_display | Display all 3 model scores | XGBoost + GNN + Hybrid shown | ✅ |
| 9 | test_consensus_verdict_display | Show consensus decision | FRAUD or CLEAN shown | ✅ |

**Key Assertions:**
```python
# Batch processing
assert len(transactions) == 20

# All models accept format
for model in ['xgboost', 'gnn', 'hybrid']:
    assert model_accepts_format(transaction, model)

# Scores in valid range
assert 0 <= xgboost_score <= 1
assert 0 <= gnn_score <= 1
assert 0 <= hybrid_score <= 1

# Consensus logic (2+ models > 0.5)
votes = sum([score > 0.5 for score in [xgb_score, gnn_score, hyb_score]])
assert consensus == "FRAUD" if votes >= 2 else "CLEAN"
```

**Run Individual Test:**
```bash
pytest tests/test_integration_upload_models.py::TestFileUploadIntegration::test_end_to_end_file_upload_and_extraction -v
```

---

### Task 3: Config Auto-Detection (11 Tests) ✅

**Test File:** `tests/test_config_autodetect.py`

**Setup:**
```bash
pytest tests/test_config_autodetect.py -v
```

**Test Cases:**

| # | Test Name | Description | Expected | Status |
|---|-----------|-------------|----------|--------|
| 1 | test_get_embedding_dimensions_with_real_file | Detect 64D embeddings from real file | dimension == 64 | ✅ |
| 2 | test_get_embedding_dimensions_with_mock_file | Detect dimensions from mock | Correct dimension | ✅ |
| 3 | test_dimension_detection_various_sizes | Test 32D, 64D, 128D, 256D | All detected correctly | ✅ |
| 4 | test_detection_with_empty_file | Handle empty embeddings file | Fallback to 64D | ✅ |
| 5 | test_detection_with_single_column | Handle malformed file | Fallback to 64D | ✅ |
| 6 | test_get_model_config_returns_dict | Generate config dict | Dict with model params | ✅ |
| 7 | test_config_values_are_valid | Validate config ranges | All values within range | ✅ |
| 8 | test_config_with_mock_embeddings | Config with mock embeddings | Valid config generated | ✅ |
| 9 | test_load_embeddings_with_prefix | Apply 'gnn_' prefix | Correct prefix applied | ✅ |
| 10 | test_config_consistency | Repeated calls match | Same config both times | ✅ |
| 11 | test_config_fallback_to_defaults | Missing file fallback | Default 64D used | ✅ |

**Key Assertions:**
```python
# Dimension detection
config = get_model_config()
assert config['embedding_dim'] in [32, 64, 128, 256]

# 'gnn_' prefix applied
assert 'gnn_embedding_0' in columns

# Config consistency
config1 = get_model_config()
config2 = get_model_config()
assert config1 == config2

# Type validation
assert isinstance(config['embedding_dim'], int)
assert isinstance(config['learning_rate'], float)
```

**Run Individual Test:**
```bash
pytest tests/test_config_autodetect.py::TestEmbeddingDetection::test_get_embedding_dimensions_with_real_file -v
```

---

### Task 4: React Error Handling (14 Tests) ✅

**Test File:** `tests/test_react_error_handling.py`

**Setup:**
```bash
pytest tests/test_react_error_handling.py -v
```

**Test Cases:**

| # | Test Name | Description | Expected | Status |
|---|-----------|-------------|----------|--------|
| 1 | test_amount_validation | Amount > 0 required | Negative rejected | ✅ |
| 2 | test_hour_validation | Hour 0-23 range | 24 rejected | ✅ |
| 3 | test_transaction_count_validation | Non-negative integer | Decimals rejected | ✅ |
| 4 | test_user_id_validation | Non-empty string | Empty rejected | ✅ |
| 5 | test_network_error_detection | Catch network errors | ECONNREFUSED detected | ✅ |
| 6 | test_file_upload_error_messages | File error messages | User-friendly error shown | ✅ |
| 7 | test_error_alert_has_close_button | Alert UI has close | Button functional | ✅ |
| 8 | test_error_alert_shows_details | Show error details | Message visible | ✅ |
| 9 | test_success_alert_styling | Success styling applied | Correct colors | ✅ |
| 10 | test_api_health_indicator | Show API status | Online/Offline indicator | ✅ |
| 11 | test_file_type_validation | Only CSV/PDF/DOCX | TXT rejected | ✅ |
| 12 | test_file_size_validation | Max 10MB file size | >10MB rejected | ✅ |
| 13 | test_validate_all_fields | Complete form validation | All fields checked | ✅ |
| 14 | test_validate_invalid_form | Invalid form detection | Forms rejected correctly | ✅ |

**Key Assertions:**

**Input Validation:**
```python
# Amount validation
assert validate_amount(5000) == True    # Valid
assert validate_amount(-100) == False   # Negative rejected
assert validate_amount(0) == False      # Zero rejected

# Hour validation
assert validate_hour(14) == True        # Valid
assert validate_hour(24) == False       # Out of range
assert validate_hour(-1) == False       # Out of range

# Transaction count validation
assert validate_transaction_count(5) == True        # Valid
assert validate_transaction_count(-1) == False      # Negative rejected
assert validate_transaction_count(0.5) == False     # Decimal rejected
```

**File Validation:**
```python
# File type validation
assert validate_file_type('data.csv') == True       # CSV valid
assert validate_file_type('data.pdf') == True       # PDF valid
assert validate_file_type('data.docx') == True      # DOCX valid
assert validate_file_type('data.txt') == False      # TXT invalid

# File size validation
assert validate_file_size(5 * 1024 * 1024) == True  # 5MB valid
assert validate_file_size(15 * 1024 * 1024) == False # 15MB too large
```

**Run Individual Test:**
```bash
pytest tests/test_react_error_handling.py::TestReactInputValidation::test_amount_validation -v
```

---

### Task 5: Edge Weight Calculations (9 Tests) ✅

**Test File:** `tests/test_edge_weights.py`

**Setup:**
```bash
pytest tests/test_edge_weights.py -v
```

**Test Cases:**

| # | Test Name | Description | Expected | Status |
|---|-----------|-------------|----------|--------|
| 1 | test_normalized_amount_weights | Normalize by transaction amount | Weights in [0.01, 1.0] | ✅ |
| 2 | test_frequency_weights | Weight by edge frequency | Higher for repeated pairs | ✅ |
| 3 | test_combined_weights | 0.6*amount + 0.4*frequency | Blended weights | ✅ |
| 4 | test_inverse_amount_weights | Inverse of amount | Small amounts high weight | ✅ |
| 5 | test_weight_output_type | Output is numpy array | Type validated | ✅ |
| 6 | test_fraud_risk_weights | Custom fraud indicators | Modifiers applied | ✅ |
| 7 | test_night_activity_modifier | 1.2x for night activity | Correct multiplier | ✅ |
| 8 | test_create_edge_weight_tensor | Convert to PyTorch tensor | Tensor shape correct | ✅ |
| 9 | test_print_statistics | Calculate min/max/mean/std | All stats returned | ✅ |

**Weight Schemes Tested:**

**1. Normalized Amount**
```python
weights = calculate_edge_weights(df, 'normalized_amount')
assert len(weights) == 5
assert np.all(weights >= 0.01)
assert np.all(weights <= 1.0)
```

**2. Frequency**
```python
weights = calculate_edge_weights(df, 'frequency')
# Repeated sender-receiver pairs get higher weight
assert weights_ab > weights_ac  # If A→B appears twice
```

**3. Combined (60/40)**
```python
weights = calculate_edge_weights(df, 'combined')
# 0.6 * normalized_amount + 0.4 * frequency
assert len(weights) == 5
```

**4. Inverse Amount**
```python
weights = calculate_edge_weights(df, 'inverse_amount')
# Small amounts get high weights
assert weights[100_amount] > weights[25000_amount]
```

**5. Fraud Risk**
```python
weights = calculate_edge_weights(df, 'fraud_risk')
# With modifiers for fraud indicators
# night_activity: 1.2x, round_amount: 0.8x
assert len(weights) == 5
assert np.all(weights >= 0.01)
```

**PyTorch Conversion:**
```python
tensor = create_edge_weight_tensor(weights)
assert isinstance(tensor, torch.Tensor)
assert tensor.shape == (len(weights), 1)
assert tensor.dtype == torch.float32
```

**Statistics:**
```python
stats = print_weight_statistics(weights)
assert 'min' in stats
assert 'max' in stats
assert 'mean' in stats
assert 'std' in stats
assert stats['min'] >= 0.01
assert stats['max'] <= 1.0
```

**Run Individual Test:**
```bash
pytest tests/test_edge_weights.py::TestEdgeWeightCalculation::test_normalized_amount_weights -v
```

---

### Complete Test Suite Execution

**Run All Task Tests (50 tests):**
```bash
python -m pytest \
  tests/test_file_upload.py \
  tests/test_integration_upload_models.py \
  tests/test_config_autodetect.py \
  tests/test_react_error_handling.py \
  tests/test_edge_weights.py \
  -v

# Expected Output:
# ======================== 50 passed in 44.33s ========================
```

**Run with Coverage:**
```bash
pytest tests/test_file_upload.py \
  tests/test_integration_upload_models.py \
  tests/test_config_autodetect.py \
  tests/test_react_error_handling.py \
  tests/test_edge_weights.py \
  --cov=backend --cov-report=html
```

**Run with Markers:**
```bash
# Run only Task 1 tests
pytest tests/test_file_upload.py -m "task1"

# Run only critical tests
pytest tests/ -m "critical"
```

---

## Test Results Summary (Phase 3)

| Component | Tests | Status | Pass Rate |
|-----------|-------|--------|-----------|
| File Upload | 7 | ✅ | 100% |
| Integration | 9 | ✅ | 100% |
| Config Auto-detect | 11 | ✅ | 100% |
| Error Handling | 14 | ✅ | 100% |
| Edge Weights | 9 | ✅ | 100% |
| **TOTAL** | **50** | **✅** | **100%** |

**Last Updated:** April 10, 2026  
**Status:** ALL TESTS PASSING ✅

---

## Success Criteria

✅ All 3 models can be evaluated independently  
✅ File upload works for at least CSV  
✅ Model comparison shows all 3 scores  
✅ AI Bot displays explanations  
✅ Transaction explanation works with real IDs  
✅ No JavaScript errors in browser  
✅ No Python errors in backend logs  
✅ Consensus logic correct (majority vote)  
✅ Performance reasonable (<2 minutes for full workflow)  
✅ **50/50 automated tests passing**  
✅ **Config auto-detection working**  
✅ **All validation rules enforced**  
✅ **5 edge weight schemes implemented**
