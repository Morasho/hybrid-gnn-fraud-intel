import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

print("Group 15: STACKED HYBRID GNN-XGBoost Evaluation ")

# 1. Load the Data
print("Loading Tabular features...")
df = pd.read_csv('data/processed/final_model_data.csv')

print("Loading GNN Probabilities (The Stacked Feature)...")
probs_df = pd.read_csv('data/processed/gnn_probabilities.csv')

# 2. STACKING
hybrid_df = pd.concat([df, probs_df], axis=1)

# 3. Prepare for Machine Learning
drop_cols = ['sender_id', 'receiver_id', 'timestamp', 'device_id', 'agent_id', 
             'is_fraud', 'fraud_scenario']
X = hybrid_df.drop(columns=drop_cols, errors='ignore')
y = hybrid_df['is_fraud']
scenarios = hybrid_df['fraud_scenario']

# 4. Split Data (Strict 42 Seed)
X_train, X_test, y_train, y_test, scen_train, scen_test = train_test_split(
    X, y, scenarios, test_size=0.2, random_state=42, stratify=y
)

# 5. Train the TUNED Stacked XGBoost
print(f"Training TUNED STACKED XGBoost on {len(X_train)} transactions...")
pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

stacked_model = XGBClassifier(
    n_estimators=150,           
    max_depth=4,                
    learning_rate=0.05,         
    colsample_bytree=0.6,       
    scale_pos_weight=pos_weight * 1.5, 
    random_state=42,
    eval_metric='logloss'
)
stacked_model.fit(X_train, y_train)

# 6. Evaluation: PURE ML PERFORMANCE (Threshold = 0.50)
print("\n STACKED Model Detection Analysis ")
predictions = stacked_model.predict(X_test)
probabilities = stacked_model.predict_proba(X_test)[:, 1]

results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': predictions,
    'Scenario': scen_test
})

actual_fraud = results[results['Actual'] == 1]

print(f"{'Fraud Topology':<20} | {'Caught (True Pos)'} | {'Missed (False Neg)'} | {'Recall (Detection Rate)'}")
print("-" * 75)

for scenario in actual_fraud['Scenario'].unique():
    scenario_data = actual_fraud[actual_fraud['Scenario'] == scenario]
    total_cases = len(scenario_data)
    caught = sum(scenario_data['Predicted'] == 1)
    missed = total_cases - caught
    recall = (caught / total_cases) * 100 if total_cases > 0 else 0.0
    print(f"{scenario:<20} | {caught:<17} | {missed:<18} | {recall:.1f}%")

print("\nOverall Performance:")
print(classification_report(y_test, predictions, target_names=['Safe (0)', 'Fraud (1)']))
print(f"STACKED ROC-AUC Score: {roc_auc_score(y_test, probabilities):.4f}")

# 7. Evaluation: THE TRAFFIC LIGHT SYSTEM (Human-in-the-Loop)
print("\nSTACKED Model: Business Logic")
business_decisions = []
for prob in probabilities:
    if prob >= 0.85:
        business_decisions.append('AUTO_FREEZE')
    elif prob >= 0.25:
        business_decisions.append('MANUAL_REVIEW')
    else:
        business_decisions.append('SAFE')

results_biz = pd.DataFrame({
    'Actual': y_test,
    'Probability': probabilities,
    'Decision': business_decisions
})

actual_fraud_biz = results_biz[results_biz['Actual'] == 1]
total_fraud = len(actual_fraud_biz)

auto_caught = len(actual_fraud_biz[actual_fraud_biz['Decision'] == 'AUTO_FREEZE'])
analyst_caught = len(actual_fraud_biz[actual_fraud_biz['Decision'] == 'MANUAL_REVIEW'])
missed_fraud = len(actual_fraud_biz[actual_fraud_biz['Decision'] == 'SAFE'])

print(f"Total Actual Fraud Cases in Test Set: {total_fraud}")
print("-" * 65)
print(f" AUTO-FREEZE (High Precision): {auto_caught} cases caught instantly.")
print(f" ANALYST QUEUE (High Recall) : {analyst_caught} cases sent to human review.")
print(f" MISSED (False Negatives)    : {missed_fraud} cases escaped.")
print("-" * 65)

system_recall = ((auto_caught + analyst_caught) / total_fraud) * 100
print(f"Total SYSTEM Recall (Model + Analyst): {system_recall:.1f}%")

safe_in_review = len(results_biz[(results_biz['Actual'] == 0) & (results_biz['Decision'] == 'MANUAL_REVIEW')])
print(f"\nAnalyst Workload: There are {safe_in_review} innocent transactions mixed into the Review Queue.")
print("The human analyst acts as the ultimate filter to protect Precision!")

# 8. Tier 2 Autonomous Agent: AI Fraud Analyst
print("\n TIER 2: AI Fraud Analyst")

# 1. The Agent intercepts the Manual Review Queue
review_indices = results_biz[results_biz['Decision'] == 'MANUAL_REVIEW'].index
review_data = X_test.loc[review_indices].copy()
review_data['Probability'] = results_biz.loc[review_indices, 'Probability']
review_data['Actual'] = results_biz.loc[review_indices, 'Actual']

# 2. The Agent's Business Logic Brain
def ai_analyst_logic(row):
    prob = row['Probability']
    # Safely grab the amount (defaults to 500 if the column isn't found)
    amount = row.get('amount', row.get('Amount', 500)) 
    
    # Safely grab transaction velocity (defaults to 1 if not found)
    # Change 'transactions_last_24hr' to whatever your actual column name is!
    velocity = row.get('transactions_last_24hr', 1) 

    # RULE 1: The "Micro-Scam" Check (Your 50 Ksh scam scenario!)
    # If XGBoost is highly suspicious (>60%) and the amount is tiny with high velocity
    if prob > 0.60 and amount < 200 and velocity > 2:
        return 'CONFIRMED_FRAUD'
        
    # RULE 2: The "Safe Normalcy" Check
    # If XGBoost is barely suspicious (<40%) and it's a completely normal amount
    elif prob < 0.40 and amount > 300 and amount < 5000:
        return 'AUTO_CLEARED_SAFE'
        
    # RULE 3: The "High Risk Value" Check
    # If the amount is massive, the AI refuses to clear it. A human MUST look.
    elif amount > 50000:
        return 'REQUIRE_HUMAN'
        
    # Default: If the AI is confused, keep it in the queue for the human analyst
    else:
        return 'REQUIRE_HUMAN'

# 3. The Agent processes the thousands of tickets instantly
print("AI Agent is reviewing transaction histories...")
review_data['AI_Decision'] = review_data.apply(ai_analyst_logic, axis=1)

# 4. Calculate the Business Savings
original_queue_size = len(review_data)
auto_cleared = len(review_data[review_data['AI_Decision'] == 'AUTO_CLEARED_SAFE'])
ai_caught = len(review_data[review_data['AI_Decision'] == 'CONFIRMED_FRAUD'])
remaining_human = len(review_data[review_data['AI_Decision'] == 'REQUIRE_HUMAN'])

# Let's see if the AI accidentally cleared any actual fraudsters
accidental_clearances = len(review_data[(review_data['AI_Decision'] == 'AUTO_CLEARED_SAFE') & (review_data['Actual'] == 1)])

print(f"\nOriginal Human Queue Size : {original_queue_size} tickets")
print("-" * 65)
print(f" AI Auto-Cleared (Safe)   : {auto_cleared} false alarms removed instantly.")
print(f" AI Confirmed Fraud       : {ai_caught} tricky fraudsters locked down.")
print(f" Remaining Human Workload : {remaining_human} tickets left for the real humans.")
print("-" * 65)
print(f"-> The AI Agent reduced the human workload by {(auto_cleared/original_queue_size)*100:.1f}%!")
print(f"-> Accidental Escapes (Failed clears): {accidental_clearances} fraudsters slipped past the AI.")