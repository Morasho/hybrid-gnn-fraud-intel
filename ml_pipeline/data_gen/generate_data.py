import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta
import random
import os

# Create directory
os.makedirs('data/raw', exist_ok=True)

# Configuration
NUM_USERS = 3000
NUM_AGENTS = 200
NUM_DEVICES = 2500
NUM_TRANSACTIONS = 15000
START_DATE = datetime(2026, 3, 1)

print("1. Generating Entities (Nodes)...")

# Generate Devices
devices = pd.DataFrame({
    'device_id': [f"D_{uuid.uuid4().hex[:6]}" for _ in range(NUM_DEVICES)],
    'is_rooted': np.random.choice([True, False], size=NUM_DEVICES, p=[0.05, 0.95])
})

# Generate Users
users = pd.DataFrame({
    'user_id': [f"U_{uuid.uuid4().hex[:6]}" for _ in range(NUM_USERS)],
    'account_age_days': np.random.randint(1, 1800, size=NUM_USERS),
    'kyc_level': np.random.choice(['Tier_1', 'Tier_2', 'Tier_3'], size=NUM_USERS, p=[0.2, 0.5, 0.3]),
    'has_defaulted': 0, # Default to 0, updated in Case 4
    'device_id': np.random.choice(devices['device_id'], size=NUM_USERS)
})

# Generate Agents
agents = pd.DataFrame({
    'agent_id': [f"A_{uuid.uuid4().hex[:6]}" for _ in range(NUM_AGENTS)],
    'agent_type': np.random.choice(['Cash_Agent', 'Business_Till'], size=NUM_AGENTS, p=[0.6, 0.4]),
    'location': np.random.choice(['Nairobi', 'Juja', 'Kiambu', 'Mombasa', 'Nakuru'], size=NUM_AGENTS)
})

# Generate Institutions
institutions = pd.DataFrame({
    'institution_id': ['I_FULIZA', 'I_MSHWARI'],
    'name': ['Fuliza', 'M-Shwari']
})

print("2. Generating Normal Background Transactions...")
transactions = []

def add_tx(sender, receiver, tx_type, amount, timestamp, is_fraud=0, scenario='Normal'):
    transactions.append({
        'transaction_id': f"TX_{uuid.uuid4().hex[:8]}",
        'sender_id': sender,
        'receiver_id': receiver,
        'tx_type': tx_type,
        'amount': round(amount, 2),
        'timestamp': timestamp,
        'is_fraud': is_fraud,
        'fraud_scenario': scenario
    })

cash_agents = agents[agents['agent_type'] == 'Cash_Agent']['agent_id'].tolist()
business_tills = agents[agents['agent_type'] == 'Business_Till']['agent_id'].tolist()
user_ids = users['user_id'].tolist()

# Generate normal noise
for _ in range(NUM_TRANSACTIONS):
    tx_type = np.random.choice(['P2P_TRANSFER', 'PAYMENT', 'WITHDRAWAL', 'LOAN_DISBURSEMENT'], p=[0.4, 0.3, 0.2, 0.1])
    sender = random.choice(user_ids)
    tx_time = START_DATE + timedelta(minutes=random.randint(1, 20000))
    amount = random.uniform(50, 15000)
    
    if tx_type == 'P2P_TRANSFER':
        add_tx(sender, random.choice(user_ids), tx_type, amount, tx_time)
    elif tx_type == 'PAYMENT':
        add_tx(sender, random.choice(business_tills), tx_type, amount, tx_time)
    elif tx_type == 'WITHDRAWAL':
        add_tx(sender, random.choice(cash_agents), tx_type, amount, tx_time)
    elif tx_type == 'LOAN_DISBURSEMENT':
        add_tx(random.choice(['I_FULIZA', 'I_MSHWARI']), sender, tx_type, amount, tx_time)

print("3. Injecting Topologically Complex Fraud Rings...")

#  Case 1: Agent Reversal Scam (Cycle -> Fan-in -> Reversal) 
c1_users = random.sample(user_ids, 5) # A, B, C, D, E
c1_time = START_DATE + timedelta(days=2)
# Layering (Cycle)
add_tx(c1_users[0], c1_users[1], 'P2P_TRANSFER', 40000, c1_time, 1, 'Case_1_Reversal')
add_tx(c1_users[1], c1_users[2], 'P2P_TRANSFER', 39500, c1_time + timedelta(minutes=5), 1, 'Case_1_Reversal')
add_tx(c1_users[2], c1_users[3], 'P2P_TRANSFER', 39000, c1_time + timedelta(minutes=10), 1, 'Case_1_Reversal')
# Fan-in & Cashout
add_tx(c1_users[3], c1_users[4], 'P2P_TRANSFER', 38500, c1_time + timedelta(minutes=15), 1, 'Case_1_Reversal')
add_tx(c1_users[4], random.choice(cash_agents), 'WITHDRAWAL', 38000, c1_time + timedelta(minutes=20), 1, 'Case_1_Reversal')
# Reversal Request from Victim A against B
add_tx(c1_users[0], c1_users[1], 'REVERSAL_REQUEST', 40000, c1_time + timedelta(minutes=25), 1, 'Case_1_Reversal')

#  Case 2: Mule Accounts & SIM Swap (Shared Device Star Topology) 
coordinator = random.choice(user_ids)
# Select 10 users to act as mules and force them onto the SAME device (synthetic identity clustering)
mules = random.sample(user_ids, 10)
shared_device = "D_FRAUD_999"
users.loc[users['user_id'].isin(mules), 'device_id'] = shared_device
users.loc[users['user_id'].isin(mules), 'account_age_days'] = random.randint(1, 3) # Brand new accounts

c2_time = START_DATE + timedelta(days=5)
for mule in mules:
    # Mules send to coordinator
    add_tx(mule, coordinator, 'P2P_TRANSFER', random.uniform(5000, 10000), c2_time + timedelta(minutes=random.randint(1,60)), 1, 'Case_2_Mule_SIM_Swap')

#  Case 3: Stolen Identity to Fast Cash-out Explosion 
victim = random.choice(user_ids)
fast_mules = random.sample(user_ids, 5)
c3_time = START_DATE + timedelta(days=7)

for mule in fast_mules:
    # Explosive star: within 60 seconds
    tx_time = c3_time + timedelta(seconds=random.randint(1, 60))
    add_tx(victim, mule, 'P2P_TRANSFER', 15000, tx_time, 1, 'Case_3_Fast_Cashout')
    # Immediate withdrawal within 2 minutes of receiving
    add_tx(mule, random.choice(cash_agents), 'WITHDRAWAL', 14900, tx_time + timedelta(seconds=random.randint(60, 120)), 1, 'Case_3_Fast_Cashout')

#  Case 4: Synecdoche Circles (Loan Fraud / Homophily) 
loan_fraudsters = random.sample(user_ids, 8)
c4_time = START_DATE + timedelta(days=10)

# Create dense covert community (Homophily)
for _ in range(25):
    add_tx(random.choice(loan_fraudsters), random.choice(loan_fraudsters), 'P2P_TRANSFER', random.uniform(500, 2000), c4_time + timedelta(hours=random.randint(1, 48)), 1, 'Case_4_Loan_Fraud')

# Borrow and vanish
for fraudster in loan_fraudsters:
    add_tx('I_FULIZA', fraudster, 'LOAN_DISBURSEMENT', random.uniform(5000, 20000), c4_time + timedelta(hours=50), 1, 'Case_4_Loan_Fraud')
    # Update Node attribute to reflect default
    users.loc[users['user_id'] == fraudster, 'has_defaulted'] = 1

#  Case 5: Fraudulent Business Till Transactions (Densification) 
corrupt_till = random.choice(business_tills)
till_washers = random.sample(user_ids, 3)
c5_time = START_DATE + timedelta(days=12)

# Inflate volume rapidly over 2 days
for _ in range(40):
    washer = random.choice(till_washers)
    tx_time = c5_time + timedelta(hours=random.randint(1, 48))
    add_tx(washer, corrupt_till, 'PAYMENT', random.uniform(40000, 90000), tx_time, 1, 'Case_5_Till_Inflation')

# Convert to DataFrame
tx_df = pd.DataFrame(transactions)

print("4. Saving Data to /data/raw/ ...")
users.to_csv('data/raw/users.csv', index=False)
agents.to_csv('data/raw/agents.csv', index=False)
devices.to_csv('data/raw/devices.csv', index=False)
institutions.to_csv('data/raw/institutions.csv', index=False)
tx_df.to_csv('data/raw/transactions.csv', index=False)

print(f"Success! Generated {len(tx_df)} transactions spanning Normal and 5 Fraud Topologies.")