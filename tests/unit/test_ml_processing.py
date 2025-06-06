import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import tempfile


@pytest.mark.unit
@pytest.mark.requires_ml
class TestMLProcessing:
    """Unit tests for ML processing functionality"""
    
    @patch('ml_processing.SentenceTransformer')
    def test_load_model(self, mock_sentence_transformer):
        """Test loading the sentence transformer model"""
        from ml_processing import load_clip_model
        
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        model = load_clip_model()
        
        assert model is not None
        mock_sentence_transformer.assert_called_once()
    
    @patch('ml_processing.SentenceTransformer')
    def test_encode_text_query(self, mock_sentence_transformer, mock_ml_models):
        """Test text query encoding"""
        from ml_processing import encode_text_query
        
        mock_model = mock_ml_models['model']
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4] * 128)
        
        with patch('ml_processing.load_clip_model', return_value=mock_model):
            embedding = encode_text_query("test query")
            
            assert embedding is not None
            assert len(embedding) == 512  # CLIP model embedding size
            mock_model.encode.assert_called_once_with("test query")
    
    @patch('ml_processing.SentenceTransformer')
    def test_encode_image(self, mock_sentence_transformer, mock_ml_models):
        """Test image encoding"""
        from ml_processing import encode_image
        from PIL import Image
        
        mock_model = mock_ml_models['model']
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4] * 128)
        
        # Create a test image
        test_image = Image.new('RGB', (224, 224), color='red')
        
        with patch('ml_processing.load_clip_model', return_value=mock_model):
            embedding = encode_image(test_image)
            
            assert embedding is not None
            assert len(embedding) == 512
            mock_model.encode.assert_called_once()
    
    def test_extract_frames_from_video(self, mock_ffmpeg, test_media_files):
        """Test frame extraction from video"""
        from ml_processing import extract_frames_from_video
        
        video_path = test_media_files["video"]
        timestamps = [10.0, 30.0, 60.0]
        
        with patch('ml_processing.subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            # Mock PIL Image.open to return a test image
            with patch('PIL.Image.open') as mock_image_open:
                mock_image = Mock()
                mock_image_open.return_value = mock_image
                
                frames = extract_frames_from_video(video_path, timestamps)
                
                assert isinstance(frames, list)
                assert len(frames) == len(timestamps)
    
    def test_process_video_for_search(self, mock_ffmpeg, mock_ml_models, test_media_files):
        """Test processing video for semantic search"""
        from ml_processing import process_video_for_search
        
        video_path = test_media_files["video"]
        video_id = 1
        
        with patch('ml_processing.extract_frames_from_video') as mock_extract, \
             patch('ml_processing.encode_image') as mock_encode, \
             patch('ml_processing.chroma_manager') as mock_chroma:
            
            # Mock frame extraction
            mock_frames = [Mock(), Mock(), Mock()]
            mock_extract.return_value = mock_frames
            
            # Mock encoding
            mock_encode.return_value = np.array([0.1, 0.2, 0.3, 0.4] * 128)
            
            # Mock ChromaDB
            mock_chroma.add_embeddings.return_value = True
            
            result = process_video_for_search(video_path, video_id)
            
            assert result is not None
            mock_extract.assert_called_once()
            assert mock_encode.call_count == len(mock_frames)
            mock_chroma.add_embeddings.assert_called_once()


@pytest.mark.unit
@pytest.mark.requires_ml
class TestChromaDBManager:
    """Unit tests for ChromaDB manager"""
    
    @patch('ml_processing.chromadb.Client')
    def test_chroma_manager_initialization(self, mock_chromadb_client):
        """Test ChromaDB manager initialization"""
        from ml_processing import ChromaDBManager
        
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb_client.return_value = mock_client
        
        manager = ChromaDBManager()
        
        assert manager is not None
        mock_chromadb_client.assert_called_once()
        mock_client.get_or_create_collection.assert_called_once()
    
    @patch('ml_processing.chromadb.Client')
    def test_add_embeddings(self, mock_chromadb_client, mock_ml_models):
        """Test adding embeddings to ChromaDB"""
        from ml_processing import ChromaDBManager
        
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb_client.return_value = mock_client
        
        manager = ChromaDBManager()
        
        # Test data
        embeddings = [np.array([0.1, 0.2, 0.3, 0.4] * 128) for _ in range(3)]
        metadatas = [
            {"video_id": 1, "timestamp": 10.0, "frame_index": 0},
            {"video_id": 1, "timestamp": 30.0, "frame_index": 1},
            {"video_id": 1, "timestamp": 60.0, "frame_index": 2}
        ]
        
        result = manager.add_embeddings(embeddings, metadatas)
        
        assert result is True
        mock_collection.add.assert_called_once()
    
    @patch('ml_processing.chromadb.Client')
    def test_search_embeddings(self, mock_chromadb_client):
        """Test searching embeddings in ChromaDB"""
        from ml_processing import ChromaDBManager
        
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb_client.return_value = mock_client
        
        # Mock search results
        mock_collection.query.return_value = {
            'ids': [['frame_1_0', 'frame_1_1']],
            'distances': [[0.1, 0.3]],
            'metadatas': [[
                {'video_id': 1, 'timestamp': 10.0},
                {'video_id': 1, 'timestamp': 30.0}
            ]]
        }
        
        manager = ChromaDBManager()
        
        query_embedding = np.array([0.1, 0.2, 0.3, 0.4] * 128)
        results = manager.search_embeddings(query_embedding, n_results=10)
        
        assert results is not None
        assert 'ids' in results
        assert 'distances' in results
        assert 'metadatas' in results
        mock_collection.query.assert_called_once()
    
    @patch('ml_processing.chromadb.Client')
    def test_get_collection_stats(self, mock_chromadb_client):
        """Test getting collection statistics"""
        from ml_processing import ChromaDBManager
        
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 100
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb_client.return_value = mock_client
        
        manager = ChromaDBManager()
        stats = manager.get_collection_stats()
        
        assert stats is not None
        assert 'total_frames' in stats
        assert stats['total_frames'] == 100
    
    @patch('ml_processing.chromadb.Client')
    def test_delete_video_embeddings(self, mock_chromadb_client):
        """Test deleting video embeddings"""
        from ml_processing import ChromaDBManager
        
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb_client.return_value = mock_client
        
        manager = ChromaDBManager()
        
        result = manager.delete_video_embeddings(video_id=1)
        
        assert result is True
        mock_collection.delete.assert_called_once()


@pytest.mark.unit
@pytest.mark.requires_ml
class TestSemanticSearch:
    """Unit tests for semantic search functionality"""
    
    @patch('ml_processing.encode_text_query')
    @patch('ml_processing.chroma_manager')
    @patch('data_access.get_media_files_from_db')
    def test_semantic_search_videos(self, mock_get_media, mock_chroma, mock_encode):
        """Test semantic search for videos"""
        from ml_processing import semantic_search_videos
        
        # Mock text encoding
        mock_encode.return_value = np.array([0.1, 0.2, 0.3, 0.4] * 128)
        
        # Mock ChromaDB search results
        mock_chroma.search_embeddings.return_value = {
            'ids': [['frame_1_0', 'frame_2_0']],
            'distances': [[0.1, 0.3]],
            'metadatas': [[
                {'video_id': 1, 'timestamp': 10.0},
                {'video_id': 2, 'timestamp': 20.0}
            ]]
        }
        
        # Mock database results
        mock_get_media.return_value = {
            'media_files': [
                {'id_db': 1, 'filename': 'video1.mp4', 'user_title': 'Test Video 1'},
                {'id_db': 2, 'filename': 'video2.mp4', 'user_title': 'Test Video 2'}
            ]
        }
        
        results = semantic_search_videos("test query", n_results=10)
        
        assert isinstance(results, list)
        assert len(results) >= 0
        mock_encode.assert_called_once_with("test query")
        mock_chroma.search_embeddings.assert_called_once()
    
    @patch('ml_processing.encode_image')
    @patch('ml_processing.chroma_manager')
    @patch('data_access.get_media_files_from_db')
    def test_semantic_search_by_image(self, mock_get_media, mock_chroma, mock_encode):
        """Test semantic search by image"""
        from ml_processing import semantic_search_by_image
        from PIL import Image
        
        # Create test image
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # Mock image encoding
        mock_encode.return_value = np.array([0.1, 0.2, 0.3, 0.4] * 128)
        
        # Mock ChromaDB search results
        mock_chroma.search_embeddings.return_value = {
            'ids': [['frame_1_0']],
            'distances': [[0.1]],
            'metadatas': [[{'video_id': 1, 'timestamp': 10.0}]]
        }
        
        # Mock database results
        mock_get_media.return_value = {
            'media_files': [
                {'id_db': 1, 'filename': 'video1.mp4', 'user_title': 'Test Video 1'}
            ]
        }
        
        results = semantic_search_by_image(test_image, n_results=10)
        
        assert isinstance(results, list)
        mock_encode.assert_called_once_with(test_image)
        mock_chroma.search_embeddings.assert_called_once()
    
    def test_similarity_threshold_filtering(self):
        """Test similarity threshold filtering"""
        from ml_processing import filter_results_by_similarity
        
        # Mock search results with various distances
        results = {
            'ids': [['frame1', 'frame2', 'frame3']],
            'distances': [[0.1, 0.5, 0.9]],
            'metadatas': [[
                {'video_id': 1, 'timestamp': 10.0},
                {'video_id': 2, 'timestamp': 20.0},
                {'video_id': 3, 'timestamp': 30.0}
            ]]
        }
        
        # Filter with threshold 0.6 (lower distances = higher similarity)
        filtered = filter_results_by_similarity(results, similarity_threshold=0.6)
        
        # Should keep only the first two results (distances 0.1 and 0.5)
        assert len(filtered['ids'][0]) == 2
        assert len(filtered['distances'][0]) == 2
        assert len(filtered['metadatas'][0]) == 2


@pytest.mark.unit
@pytest.mark.requires_ml
class TestMLUtilities:
    """Unit tests for ML utility functions"""
    
    def test_normalize_embedding(self):
        """Test embedding normalization"""
        from ml_processing import normalize_embedding
        
        # Test embedding
        embedding = np.array([3.0, 4.0, 0.0])
        normalized = normalize_embedding(embedding)
        
        # Should be unit vector
        np.testing.assert_almost_equal(np.linalg.norm(normalized), 1.0)
        
        # Check proportions are preserved
        expected = np.array([0.6, 0.8, 0.0])
        np.testing.assert_array_almost_equal(normalized, expected)
    
    def test_calculate_similarity_score(self):
        """Test similarity score calculation"""
        from ml_processing import calculate_similarity_score
        
        # Test embeddings
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])  # Orthogonal
        emb3 = np.array([1.0, 0.0, 0.0])  # Identical
        
        # Orthogonal vectors should have low similarity
        sim_orthogonal = calculate_similarity_score(emb1, emb2)
        assert 0.0 <= sim_orthogonal <= 0.1
        
        # Identical vectors should have high similarity
        sim_identical = calculate_similarity_score(emb1, emb3)
        assert 0.9 <= sim_identical <= 1.0
    
    def test_batch_encode_images(self, mock_ml_models):
        """Test batch encoding of images"""
        from ml_processing import batch_encode_images
        from PIL import Image
        
        # Create test images
        images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='green'),
            Image.new('RGB', (224, 224), color='blue')
        ]
        
        mock_model = mock_ml_models['model']
        mock_model.encode.return_value = np.array([[0.1, 0.2] * 256] * 3)  # 3 embeddings
        
        with patch('ml_processing.load_clip_model', return_value=mock_model):
            embeddings = batch_encode_images(images)
            
            assert len(embeddings) == 3
            assert all(len(emb) == 512 for emb in embeddings)
            mock_model.encode.assert_called_once()
    
    def test_get_video_duration(self, mock_ffmpeg):
        """Test getting video duration"""
        from ml_processing import get_video_duration
        
        with patch('ml_processing.subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "120.5"
            
            duration = get_video_duration(Path("test_video.mp4"))
            
            assert duration == 120.5
            mock_run.assert_called_once()
    
    def test_generate_frame_timestamps(self):
        """Test frame timestamp generation"""
        from ml_processing import generate_frame_timestamps
        
        # Test with 60 second video, 5 frames
        timestamps = generate_frame_timestamps(60.0, num_frames=5)
        
        assert len(timestamps) == 5
        assert timestamps[0] >= 0
        assert timestamps[-1] <= 60.0
        assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
    
    def test_cleanup_temp_files(self):
        """Test temporary file cleanup"""
        from ml_processing import cleanup_temp_files
        
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create some test files
            test_files = []
            for i in range(3):
                test_file = temp_path / f"test_frame_{i}.jpg"
                test_file.write_bytes(b"fake image data")
                test_files.append(test_file)
            
            # Verify files exist
            assert all(f.exists() for f in test_files)
            
            # Clean up
            cleanup_temp_files(test_files)
            
            # Files should be removed
            assert not any(f.exists() for f in test_files)


@pytest.mark.unit
@pytest.mark.requires_ml
class TestMLErrorHandling:
    """Unit tests for ML error handling"""
    
    def test_model_loading_failure(self):
        """Test handling of model loading failure"""
        from ml_processing import load_clip_model
        
        with patch('ml_processing.SentenceTransformer', side_effect=Exception("Model load failed")):
            with pytest.raises(Exception):
                load_clip_model()
    
    def test_encoding_failure(self, mock_ml_models):
        """Test handling of encoding failure"""
        from ml_processing import encode_text_query
        
        mock_model = mock_ml_models['model']
        mock_model.encode.side_effect = Exception("Encoding failed")
        
        with patch('ml_processing.load_clip_model', return_value=mock_model):
            with pytest.raises(Exception):
                encode_text_query("test query")
    
    def test_chromadb_connection_failure(self):
        """Test handling of ChromaDB connection failure"""
        from ml_processing import ChromaDBManager
        
        with patch('ml_processing.chromadb.Client', side_effect=Exception("Connection failed")):
            with pytest.raises(Exception):
                ChromaDBManager()
    
    def test_invalid_image_handling(self):
        """Test handling of invalid images"""
        from ml_processing import encode_image
        
        # Test with None image
        with pytest.raises((ValueError, AttributeError)):
            encode_image(None)
        
        # Test with invalid image format
        with pytest.raises(Exception):
            encode_image("not_an_image")
    
    def test_video_processing_failure(self, mock_ffmpeg):
        """Test handling of video processing failure"""
        from ml_processing import extract_frames_from_video
        
        with patch('ml_processing.subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1  # FFmpeg failure
            mock_run.return_value.stderr = "FFmpeg error"
            
            with pytest.raises(Exception):
                extract_frames_from_video(Path("nonexistent.mp4"), [10.0])
    
    def test_empty_search_results(self):
        """Test handling of empty search results"""
        from ml_processing import semantic_search_videos
        
        with patch('ml_processing.encode_text_query') as mock_encode, \
             patch('ml_processing.chroma_manager') as mock_chroma:
            
            mock_encode.return_value = np.array([0.1, 0.2, 0.3, 0.4] * 128)
            
            # Mock empty search results
            mock_chroma.search_embeddings.return_value = {
                'ids': [[]],
                'distances': [[]],
                'metadatas': [[]]
            }
            
            results = semantic_search_videos("test query")
            
            assert isinstance(results, list)
            assert len(results) == 0