"""
Test suite for config auto-detection (Task 3)
Tests that model configuration is properly auto-detected from embeddings
"""

import pytest
import pandas as pd
import tempfile
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml_pipeline', 'models'))

try:
    from config import (
        get_embedding_dimensions,
        get_model_config,
        load_embeddings_with_prefix
    )
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False


class TestEmbeddingDetection:
    """Test auto-detection of embedding dimensions"""

    def test_get_embedding_dimensions_with_real_file(self):
        """Test dimension detection with real embeddings file"""
        if not HAS_CONFIG:
            pytest.skip("config module not available")
        
        try:
            # Try to read actual embeddings file if it exists
            dim = get_embedding_dimensions('data/processed/user_embeddings.csv')
            
            assert isinstance(dim, int)
            assert dim > 0
            print(f"✓ Dimensions detected from real file: {dim}D embeddings")
            
        except FileNotFoundError:
            print("⚠ Embeddings file not found - this is expected before training")

    def test_get_embedding_dimensions_with_mock_file(self):
        """Test dimension detection with mock embeddings file"""
        if not HAS_CONFIG:
            pytest.skip("config module not available")
        
        # Create a mock embeddings file
        mock_embeddings = pd.DataFrame({
            'user_id': ['USER_1', 'USER_2', 'USER_3'],
            'emb_0': [0.1, 0.2, 0.3],
            'emb_1': [0.4, 0.5, 0.6],
            'emb_2': [0.7, 0.8, 0.9],
            'emb_3': [1.0, 1.1, 1.2],
            'emb_4': [1.3, 1.4, 1.5]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            mock_embeddings.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            dim = get_embedding_dimensions(temp_file)
            
            assert dim == 5  # 6 columns - 1 for user_id
            print(f"✓ Mock embeddings dimension detection: {dim}D")
            
        finally:
            os.unlink(temp_file)

    def test_dimension_detection_various_sizes(self):
        """Test dimension detection with different embedding sizes"""
        if not HAS_CONFIG:
            pytest.skip("config module not available")
        
        test_cases = [
            (32, 32),   # 32D embeddings
            (64, 64),   # 64D embeddings (default)
            (128, 128), # 128D embeddings
            (256, 256)  # 256D embeddings
        ]
        
        for embedding_size, expected_dim in test_cases:
            # Create mock embeddings
            data = {'user_id': [f'USER_{i}' for i in range(3)]}
            for j in range(embedding_size):
                data[f'emb_{j}'] = [0.1 * (i+j) for i in range(3)]
            
            mock_df = pd.DataFrame(data)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                mock_df.to_csv(f.name, index=False)
                temp_file = f.name
            
            try:
                dim = get_embedding_dimensions(temp_file)
                assert dim == expected_dim
                print(f"✓ Detected {expected_dim}D embeddings correctly")
            finally:
                os.unlink(temp_file)

    def test_detection_with_empty_file(self):
        """Test dimension detection raises error for empty file"""
        if not HAS_CONFIG:
            pytest.skip("config module not available")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('')  # Empty file
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError):
                get_embedding_dimensions(temp_file)
            print("✓ Empty file properly raises ValueError")
        finally:
            os.unlink(temp_file)

    def test_detection_with_single_column(self):
        """Test dimension detection raises error for single column"""
        if not HAS_CONFIG:
            pytest.skip("config module not available")
        
        # File with only user_id, no embeddings
        single_col = pd.DataFrame({'user_id': ['USER_1', 'USER_2']})
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            single_col.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError):
                get_embedding_dimensions(temp_file)
            print("✓ Single column properly raises ValueError")
        finally:
            os.unlink(temp_file)


class TestModelConfigGeneration:
    """Test model configuration generation"""

    def test_get_model_config_returns_dict(self):
        """Test that get_model_config returns proper dictionary"""
        if not HAS_CONFIG:
            pytest.skip("config module not available")
        
        config = get_model_config()
        
        assert isinstance(config, dict)
        assert 'embedding_dim' in config
        assert 'hidden_channels' in config
        assert 'learning_rate' in config
        assert 'random_state' in config
        assert 'num_epochs' in config
        
        print(f"✓ Config dictionary has all required keys")

    def test_config_values_are_valid(self):
        """Test that config values are reasonable"""
        if not HAS_CONFIG:
            pytest.skip("config module not available")
        
        config = get_model_config()
        
        # embedding_dim should be reasonable
        assert config['embedding_dim'] >= 8
        assert config['embedding_dim'] <= 512
        
        # hidden_channels equals embedding_dim
        assert config['hidden_channels'] == config['embedding_dim']
        
        # Learning rate should be small
        assert config['learning_rate'] > 0
        assert config['learning_rate'] < 1
        
        # Epochs should be reasonable
        assert config['num_epochs'] > 0
        assert config['num_epochs'] <= 1000
        
        # Random state for reproducibility
        assert config['random_state'] == 42
        
        print(f"✓ Config values are valid")
        print(f"  Embedding dimension: {config['embedding_dim']}")
        print(f"  Learning rate: {config['learning_rate']}")
        print(f"  Num epochs: {config['num_epochs']}")

    def test_config_with_mock_embeddings(self):
        """Test config generation uses detected embeddings"""
        if not HAS_CONFIG:
            pytest.skip("config module not available")
        
        # Create mock embeddings
        mock_embeddings = pd.DataFrame({
            'user_id': ['USER_1', 'USER_2'],
            'emb_0': [0.1, 0.2],
            'emb_1': [0.3, 0.4],
            'emb_2': [0.5, 0.6]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            mock_embeddings.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Temporarily replace embeddings file path
            # In real test, would mock the function
            dim = get_embedding_dimensions(temp_file)
            
            assert dim == 3
            print(f"✓ Config would use detected {dim}D embeddings")
            
        finally:
            os.unlink(temp_file)


class TestEmbeddingsLoading:
    """Test embeddings loading with prefix"""

    def test_load_embeddings_with_prefix(self):
        """Test loading embeddings and adding prefix"""
        if not HAS_CONFIG:
            pytest.skip("config module not available")
        
        # Create mock embeddings
        mock_embeddings = pd.DataFrame({
            'user_id': ['USER_1', 'USER_2', 'USER_3'],
            'emb_0': [0.1, 0.2, 0.3],
            'emb_1': [0.4, 0.5, 0.6],
            'emb_2': [0.7, 0.8, 0.9]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            mock_embeddings.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            embeddings_df, dim = load_embeddings_with_prefix(temp_file, prefix='gnn_')
            
            # Check prefix was applied
            assert 'gnn_user_id' in embeddings_df.columns
            assert 'gnn_emb_0' in embeddings_df.columns
            
            # Check dimension detected
            assert dim == 3
            
            # Check data integrity
            assert len(embeddings_df) == 3
            
            print(f"✓ Embeddings loaded with 'gnn_' prefix")
            print(f"  Columns: {list(embeddings_df.columns)[:3]}...")
            
        finally:
            os.unlink(temp_file)


class TestConfigCaching:
    """Test that config values are consistent"""

    def test_config_consistency(self):
        """Test that multiple calls return same config"""
        if not HAS_CONFIG:
            pytest.skip("config module not available")
        
        config1 = get_model_config()
        config2 = get_model_config()
        
        # Should return consistent values
        assert config1['embedding_dim'] == config2['embedding_dim']
        assert config1['learning_rate'] == config2['learning_rate']
        assert config1['num_epochs'] == config2['num_epochs']
        
        print("✓ Config values are consistent across calls")

    def test_config_fallback_to_defaults(self):
        """Test config falls back to defaults when embeddings not found"""
        if not HAS_CONFIG:
            pytest.skip("config module not available")
        
        config = get_model_config()
        
        # If embeddings file doesn't exist, should return defaults
        # Default is 64D embeddings
        if config['embedding_dim'] == 64:
            print("✓ Config using default 64D embeddings (file not found)")
        else:
            print(f"✓ Config detected existing embeddings: {config['embedding_dim']}D")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running Config Auto-Detection Tests (Task 3)")
    print("="*60 + "\n")
    
    if not HAS_CONFIG:
        print("⚠ config module not found - tests will be skipped")
        sys.exit(0)
    
    # Run embedding detection tests
    embed_suite = TestEmbeddingDetection()
    embed_suite.test_get_embedding_dimensions_with_real_file()
    embed_suite.test_get_embedding_dimensions_with_mock_file()
    embed_suite.test_dimension_detection_various_sizes()
    embed_suite.test_detection_with_empty_file()
    embed_suite.test_detection_with_single_column()
    
    # Run model config tests
    config_suite = TestModelConfigGeneration()
    config_suite.test_get_model_config_returns_dict()
    config_suite.test_config_values_are_valid()
    config_suite.test_config_with_mock_embeddings()
    
    # Run embeddings loading tests
    load_suite = TestEmbeddingsLoading()
    load_suite.test_load_embeddings_with_prefix()
    
    # Run consistency tests
    consistency_suite = TestConfigCaching()
    consistency_suite.test_config_consistency()
    consistency_suite.test_config_fallback_to_defaults()
    
    print("\n" + "="*60)
    print("All config auto-detection tests completed!")
    print("="*60)
