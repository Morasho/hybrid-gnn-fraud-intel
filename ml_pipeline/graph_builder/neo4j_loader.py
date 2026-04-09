import pandas as pd
from neo4j import GraphDatabase
import time

print(" Phase 1: High-Performance Neo4j Batch Loader ")

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "12345678")  # 

def batch_load_nodes(tx, query, data, batch_size=2000):
    """Loads nodes in blocks and prints real-time progress to prevent freezing."""
    total = len(data)
    for i in range(0, total, batch_size):
        batch = data[i:i + batch_size]
        tx.run(query, parameters={'batch': batch})
        # Print a progress update to the terminal!
        print(f"   -> Processed {min(i + batch_size, total)} / {total} records...")

def load_graph_data():
    print("Loading 100,000 raw transactions from CSV...")
    df = pd.read_csv('data/raw/p2p_transfers.csv')

    # 1. Extract Unique Entities
    unique_users = pd.concat([df['sender_id'], df['receiver_id']]).unique()
    users_data = [{'user_id': uid} for uid in unique_users]
    
    unique_agents = df['agent_id'].unique()
    agents_data = [{'agent_id': aid} for aid in unique_agents]
    
    unique_devices = df['device_id'].unique()
    devices_data = [{'device_id': did} for did in unique_devices]

    # 2. Prepare Edges Data
    # Convert timestamp string to a format Neo4j likes, or just keep as string for now
    edges_data = df.to_dict('records')

    driver = GraphDatabase.driver(URI, auth=AUTH)
    
    start_time = time.time()
    with driver.session() as session:
        # LOAD NODES 
        print(f"Pushing {len(users_data)} Users to Neo4j...")
        session.execute_write(batch_load_nodes, 
            "UNWIND $batch AS row MERGE (u:User {user_id: row.user_id})", 
            users_data)

        print(f"Pushing {len(agents_data)} Agents to Neo4j...")
        session.execute_write(batch_load_nodes, 
            "UNWIND $batch AS row MERGE (a:Agent {agent_id: row.agent_id})", 
            agents_data)

        print(f"Pushing {len(devices_data)} Devices to Neo4j...")
        session.execute_write(batch_load_nodes, 
            "UNWIND $batch AS row MERGE (d:Device {device_id: row.device_id})", 
            devices_data)

        #  LOAD EDGES 
        print(f"Pushing {len(edges_data)} P2P Transfers & Connections to Neo4j... (This may take a minute)")
        
        # 1. The P2P Money Transfer
        p2p_query = """
        UNWIND $batch AS row
        MATCH (sender:User {user_id: row.sender_id})
        MATCH (receiver:User {user_id: row.receiver_id})
        MERGE (sender)-[r:P2P_TRANSFER {
            amount: toFloat(row.amount),
            timestamp: row.timestamp,
            is_fraud: toInteger(row.is_fraud),
            fraud_scenario: row.fraud_scenario
        }]->(receiver)
        """
        session.execute_write(batch_load_nodes, p2p_query, edges_data)

        # 2. The Device Connection (Crucial for SIM Swap detection)
        device_query = """
        UNWIND $batch AS row
        MATCH (sender:User {user_id: row.sender_id})
        MATCH (device:Device {device_id: row.device_id})
        MERGE (sender)-[:USES]->(device)
        """
        session.execute_write(batch_load_nodes, device_query, edges_data)

        # 3. The Agent Connection (Crucial for Cash-Out detection)
        agent_query = """
        UNWIND $batch AS row
        MATCH (sender:User {user_id: row.sender_id})
        MATCH (agent:Agent {agent_id: row.agent_id})
        MERGE (sender)-[:INTERACTS_WITH]->(agent)
        """
        session.execute_write(batch_load_nodes, agent_query, edges_data)

    driver.close()
    elapsed = time.time() - start_time
    print(f"\nGraph Database successfully built in {elapsed:.2f} seconds!")

if __name__ == "__main__":
    load_graph_data()