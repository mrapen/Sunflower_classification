import unittest
import os
import sys
import torch
import numpy as np
import tempfile
import json
from unittest.mock import patch, MagicMock, mock_open
from PIL import Image
from io import BytesIO

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules to test
from data.dataset import CustomDataset, get_dls
from visualization.visualization import Visualization
from training.train_validation import TrainValidation
from training.train_summary import TrainProcessSummary
from inference.inference import inference
from analysis.analysis import analyze_model_performance, analyze_class_performance


class TestCustomDataset(unittest.TestCase):
    """Tests for the CustomDataset class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temp directory
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_dir = os.path.join(self.temp_dir, 'dataset')
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        # Create mock data
        self.imgs = np.random.rand(10, 3, 224, 224).astype(np.float32)
        self.labels = np.array([[label//2] for label in range(10)])
        
        # Save mock data
        np.save(os.path.join(self.dataset_dir, 'Sunflower_Stages.npy'), self.imgs)
        np.save(os.path.join(self.dataset_dir, 'Sunflower_Stages_Labels.npy'), self.labels)
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove temp files
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('os.path.dirname')
    def test_dataset_initialization(self, mock_dirname):
        """Test dataset initialization."""
        mock_dirname.return_value = os.path.dirname(__file__)
        
        # Initialize dataset
        dataset = CustomDataset(root=self.dataset_dir, transformations=None)
        
        # Check if dataset was initialized correctly
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.class_names), 5)
        self.assertEqual(dataset.class_counts, {0: 2, 1: 2, 2: 2, 3: 2, 4: 2})
    
    @patch('os.path.dirname')
    def test_dataset_getitem(self, mock_dirname):
        """Test __getitem__ method of the dataset."""
        mock_dirname.return_value = os.path.dirname(__file__)
        
        # Initialize dataset
        dataset = CustomDataset(root=self.dataset_dir, transformations=None)
        
        # Get an item
        item = dataset[0]
        
        # Check if item contains all necessary keys
        self.assertIn('qry_im', item)
        self.assertIn('qry_gt', item)
        self.assertIn('pos_im', item)
        self.assertIn('neg_im', item)
        self.assertIn('neg_gt', item)
        
        # Check if query and positive images belong to the same class
        self.assertEqual(item['qry_gt'], 0)
        
        # Check if negative image belongs to a different class
        self.assertNotEqual(item['qry_gt'], item['neg_gt'])
    
    @patch('os.path.dirname')
    def test_get_dls(self, mock_dirname):
        """Test the get_dls function."""
        mock_dirname.return_value = os.path.dirname(__file__)
        
        # Get dataloaders
        tr_dl, val_dl, ts_dl, classes, cls_counts = get_dls(
            root=self.dataset_dir, 
            transformations=None, 
            batch_size=2, 
            split=[0.6, 0.2, 0.2]
        )
        
        # Check if dataloaders have the correct number of batches
        self.assertEqual(len(tr_dl), 3)  # 10 * 0.6 / 2 = 3
        self.assertEqual(len(val_dl), 1)  # 10 * 0.2 / 2 = 1
        self.assertEqual(len(ts_dl), 2)  # 10 * 0.2 / 1 = 2
        
        # Check if classes and class counts are correct
        self.assertEqual(len(classes), 5)
        self.assertEqual(cls_counts, {0: 2, 1: 2, 2: 2, 3: 2, 4: 2})


class TestVisualization(unittest.TestCase):
    """Tests for the Visualization class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock dataloaders
        self.mock_batch = {
            'qry_im': torch.rand(1, 3, 224, 224),
            'pos_im': torch.rand(1, 3, 224, 224),
            'neg_im': torch.rand(1, 3, 224, 224),
            'qry_gt': torch.tensor([0]),
            'neg_gt': torch.tensor([2])
        }
        
        self.mock_dataloader = MagicMock()
        self.mock_dataloader.__iter__.return_value = [self.mock_batch]
        self.mock_dataloader.dataset = [self.mock_batch for _ in range(10)]
        
        # Classes and counts
        self.classes = {0: 'Germination', 1: 'Seedling', 2: 'Vegetative', 3: 'Bud Formation'}
        self.cls_counts = {0: 25, 1: 30, 2: 20, 3: 25}
        
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplot')
    @patch('matplotlib.pyplot.imshow')
    @patch('matplotlib.pyplot.axis')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.show')
    def test_visualization(self, mock_show, mock_title, mock_axis, mock_imshow, mock_subplot, mock_figure):
        """Test visualization method."""
        # Initialize visualization
        vis = Visualization(
            vis_datas=[self.mock_dataloader, self.mock_dataloader, self.mock_dataloader],
            n_ims=6, rows=2, cmap="rgb", cls_names=list(self.classes.keys()),
            cls_counts=self.cls_counts
        )
        
        # Mock the tn2np method
        vis.tn2np = MagicMock(return_value=np.random.rand(224, 224, 3))
        
        # Call the visualization method
        vis.visualization()
        
        # Check if necessary methods were called
        mock_figure.assert_called()
        mock_subplot.assert_called()
        mock_imshow.assert_called()
        mock_axis.assert_called()
        mock_title.assert_called()
        mock_show.assert_called()
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.bar')
    @patch('matplotlib.pyplot.show')
    def test_data_analysis(self, mock_show, mock_bar, mock_subplots):
        """Test data_analysis method."""
        # Create mock axes
        mock_ax = MagicMock()
        mock_subplots.return_value = (MagicMock(), mock_ax)
        
        # Initialize visualization
        vis = Visualization(
            vis_datas=[self.mock_dataloader, self.mock_dataloader, self.mock_dataloader],
            n_ims=6, rows=2, cmap="rgb", cls_names=list(self.classes.keys()),
            cls_counts=self.cls_counts
        )
        
        # Call the analysis method
        vis.data_analysis(self.cls_counts, "test")
        
        # Check if necessary methods were called
        mock_subplots.assert_called()
        mock_ax.bar.assert_called_once()
        mock_ax.set_xlabel.assert_called_once()
        mock_ax.set.assert_called_once()
        mock_ax.set_ylabel.assert_called_once()
        mock_ax.set_title.assert_called_once()
        mock_show.assert_called_once()


class TestTrainValidation(unittest.TestCase):
    """Tests for the TrainValidation class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock dataloaders
        self.mock_batch = {
            'qry_im': torch.rand(2, 3, 224, 224),
            'pos_im': torch.rand(2, 3, 224, 224),
            'neg_im': torch.rand(2, 3, 224, 224),
            'qry_gt': torch.tensor([0, 1]),
            'neg_gt': torch.tensor([2, 3])
        }
        
        self.mock_tr_dl = MagicMock()
        self.mock_tr_dl.__iter__.return_value = [self.mock_batch]
        self.mock_tr_dl.__len__.return_value = 5
        self.mock_tr_dl.dataset = [self.mock_batch for _ in range(10)]
        
        self.mock_val_dl = MagicMock()
        self.mock_val_dl.__iter__.return_value = [self.mock_batch]
        self.mock_val_dl.__len__.return_value = 3
        self.mock_val_dl.dataset = [self.mock_batch for _ in range(6)]
        
        # Classes and device
        self.classes = {0: 'Germination', 1: 'Seedling', 2: 'Vegetative', 3: 'Bud Formation'}
        self.device = torch.device('cpu')
        
        # Create temp directory for saving models
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temp files
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('timm.create_model')
    @patch('torch.save')
    def test_train_validation_init(self, mock_save, mock_create_model):
        """Test initialization and basic functionality of TrainValidation."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.rand(3, 3, requires_grad=True)]
        mock_model.to.return_value = mock_model
        mock_model.forward_features.return_value = torch.rand(2, 512, 7, 7)
        mock_model.forward_head.return_value = torch.rand(2, 4)
        mock_model.blocks = [MagicMock()]
        
        mock_create_model.return_value = mock_model
        
        # Initialize trainer with dev_mode=True to run only 1 epoch
        trainer = TrainValidation(
            model_name='efficientnet_b0',
            tr_dl=self.mock_tr_dl,
            val_dl=self.mock_val_dl,
            classes=self.classes,
            device=self.device,
            lr=3e-4,
            save_dir=self.temp_dir,
            data_name='test',
            bs=2,
            run_name='test_run',
            epochs=2,
            patience=3,
            dev_mode=True
        )
        
        # Get stats
        stats = trainer.get_stats()
        
        # Check if stats have been collected
        self.assertEqual(len(stats), 12)  # 12 lists of metrics
        
        # Check if model has been saved
        mock_save.assert_called_once()


class TestTrainSummary(unittest.TestCase):
    """Tests for the TrainProcessSummary class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock training metrics
        self.tr_losses = [0.5, 0.4, 0.3, 0.25, 0.2]
        self.val_losses = [0.55, 0.45, 0.4, 0.35, 0.32]
        self.tr_accs = [0.7, 0.75, 0.8, 0.85, 0.9]
        self.val_accs = [0.65, 0.7, 0.75, 0.78, 0.8]
        self.tr_f1s = [0.75, 0.8, 0.85, 0.87, 0.9]
        self.val_f1s = [0.7, 0.75, 0.78, 0.8, 0.82]
        self.tr_specs = [0.8, 0.85, 0.87, 0.9, 0.92]
        self.val_spec = [0.75, 0.8, 0.82, 0.85, 0.87]
        self.tr_sens = [0.78, 0.82, 0.85, 0.87, 0.9]
        self.val_sens = [0.73, 0.78, 0.8, 0.83, 0.85]
        self.tr_times = [50, 48, 47, 46, 45]
        self.vl_times = [10, 10, 10, 10, 10]
        
        # Create temp directory for saving plots
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temp files
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.xticks')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_train_summary(self, mock_savefig, mock_show, mock_legend, mock_xticks, 
                          mock_title, mock_ylabel, mock_xlabel, mock_plot, mock_figure):
        """Test TrainProcessSummary functionality."""
        # Initialize summary
        summary = TrainProcessSummary(
            tr_losses=self.tr_losses,
            val_losses=self.val_losses,
            tr_accs=self.tr_accs,
            val_accs=self.val_accs,
            tr_f1s=self.tr_f1s,
            val_f1s=self.val_f1s,
            tr_specs=self.tr_specs,
            val_spec=self.val_spec,
            tr_sens=self.tr_sens,
            val_sens=self.val_sens,
            tr_times=self.tr_times,
            vl_times=self.vl_times,
            data_name='test',
            save_path=self.temp_dir
        )
        
        # Check if necessary methods were called
        self.assertEqual(mock_figure.call_count, 8) 
        self.assertEqual(mock_plot.call_count, 10)
        self.assertEqual(mock_savefig.call_count, 6)


class TestInference(unittest.TestCase):
    """Tests for the inference module."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock model and dataloader
        self.mock_model = MagicMock()
        self.mock_model.forward_features.return_value = torch.rand(1, 512, 7, 7)
        self.mock_model.forward_head.return_value = torch.rand(1, 4)
        self.mock_model.blocks = [MagicMock()]
        
        self.mock_batch = {
            'qry_im': torch.rand(1, 3, 224, 224),
            'pos_im': torch.rand(1, 3, 224, 224),
            'neg_im': torch.rand(1, 3, 224, 224),
            'qry_gt': torch.tensor([0]),
            'neg_gt': torch.tensor([2])
        }
        
        self.mock_dataloader = MagicMock()
        self.mock_dataloader.__iter__.return_value = [self.mock_batch]
        self.mock_dataloader.__len__.return_value = 5
        
        # Classes
        self.classes = ['Germination', 'Seedling', 'Vegetative', 'Bud Formation']
        
        # Device
        self.device = torch.device('cpu')
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplot')
    @patch('matplotlib.pyplot.imshow')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.axis')
    @patch('matplotlib.pyplot.show')
    @patch('pytorch_grad_cam.GradCAMPlusPlus')
    def test_inference(self, mock_gradcam, mock_show, mock_axis, mock_title, 
                      mock_imshow, mock_subplot, mock_figure):
        """Test inference function."""
        # Mock GradCAM class
        mock_gradcam_instance = MagicMock()
        mock_gradcam_instance.return_value = np.random.rand(7, 7)
        mock_gradcam.return_value = mock_gradcam_instance
        
        # Import the actual inference function
        from inference.inference import inference
        
        # Call inference
        with patch('torch.nn.functional.softmax', return_value=torch.tensor([[0.1, 0.2, 0.3, 0.4]])):
            with patch('torch.max', return_value=(torch.tensor([0.4]), torch.tensor([3]))):
                inference(
                    model=self.mock_model,
                    device=self.device,
                    test_dl=self.mock_dataloader,
                    num_ims=2,
                    row=1,
                    cls_names=self.classes,
                    im_size=224
                )
        
        # Check if necessary methods were called
        mock_figure.assert_called()
        mock_subplot.assert_called()
        mock_imshow.assert_called()
        mock_title.assert_called()
        mock_axis.assert_called()
        mock_show.assert_called()


class TestAnalysis(unittest.TestCase):
    """Tests for the analysis module."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock model and data
        self.mock_model = MagicMock()
        self.mock_model.eval.return_value = None
        self.mock_model.to.return_value = self.mock_model
        
        with patch('torch.nn.functional.softmax', return_value=torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])):
            with patch('torch.max', return_value=(torch.tensor([0.4, 0.4]), torch.tensor([3, 0]))):
                self.mock_model.return_value = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
        
        self.mock_batch = {
            'qry_im': torch.rand(2, 3, 224, 224),
            'pos_im': torch.rand(2, 3, 224, 224),
            'neg_im': torch.rand(2, 3, 224, 224),
            'qry_gt': torch.tensor([0, 1]),
            'neg_gt': torch.tensor([2, 3])
        }
        
        self.mock_dataloader = MagicMock()
        self.mock_dataloader.__iter__.return_value = [self.mock_batch]
        self.mock_dataloader.__len__.return_value = 1
        
        # Classes
        self.classes = {0: 'Germination', 1: 'Seedling', 2: 'Vegetative', 3: 'Bud Formation'}
        
        # Device
        self.device = torch.device('cpu')
        
        # Mock some methods for analyze_model_performance
        self.patches = [
            patch('torch.no_grad', return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())),
            patch('torch.nn.functional.softmax', return_value=torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])),
            patch('torch.max', return_value=(torch.tensor([0.4, 0.4]), torch.tensor([3, 0])))
        ]
        
        for p in self.patches:
            p.start()
    
    def tearDown(self):
        """Clean up after tests."""
        for p in self.patches:
            p.stop()
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.colorbar')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.yticks')
    @patch('matplotlib.pyplot.xticks')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.imshow')
    @patch('matplotlib.pyplot.figure')
    @patch('sklearn.metrics.confusion_matrix', return_value=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    @patch('sklearn.metrics.classification_report', return_value="Report")
    @patch('sklearn.metrics.accuracy_score', return_value=0.75)
    def test_analyze_model_performance(self, mock_accuracy, mock_report, mock_cm,
                                       mock_fig, mock_imshow, mock_title,
                                       mock_xticks, mock_yticks, mock_tight,
                                       mock_xlabel, mock_ylabel, mock_colorbar,
                                       mock_show):
        """Test analyze_model_performance function."""
        # Call analyze_model_performance
        all_preds, all_labels, accuracy = analyze_model_performance(
            model=self.mock_model,
            test_dl=self.mock_dataloader,
            device=self.device,
            classes=self.classes
        )
        
        # Check results
        self.assertIsInstance(all_preds, list)
        self.assertIsInstance(all_labels, list)
        self.assertEqual(accuracy, 0.75)
        mock_accuracy.assert_called_once()
        mock_cm.assert_called_once()
        mock_report.assert_called_once()
        mock_fig.assert_called()
        self.assertGreaterEqual(mock_fig.call_count, 1)
        mock_imshow.assert_called_once()
        mock_title.assert_called_once()
        mock_xticks.assert_called_once()
        mock_yticks.assert_called_once()
        mock_tight.assert_called_once()
        mock_xlabel.assert_called_once()
        mock_ylabel.assert_called_once()
        mock_colorbar.assert_called_once()
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.bar')
    @patch('matplotlib.pyplot.show')
    def test_analyze_class_performance(self, mock_show, mock_bar, mock_figure):
        all_labels = [0,1,2,3]
        all_preds  = [0,1,2,3]
        analyze_class_performance(all_labels, all_preds, self.classes)
        mock_figure.assert_called()
        self.assertGreaterEqual(mock_figure.call_count, 1)
        mock_bar.assert_called_once()
        mock_show.assert_called_once()


if __name__ == '__main__':
    unittest.main()
