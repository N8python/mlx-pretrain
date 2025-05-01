import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import yaml
import mlx.optimizers as optim
import mlx_optimizers as optim_x
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime
import os
#from mlx_lm.models.llama import Model, ModelArgs
import importlib
from mlx.utils import tree_flatten, tree_map, tree_unflatten
import inspect

def filter_valid_args(cls, arg_dict):
    valid_params = inspect.signature(cls).parameters
    return {k: v for k, v in arg_dict.items() if k in valid_params}


@dataclass
class DataConfig:
    input_file: str
    preprocessing: Dict[str, int]
    tokenizer: Dict[str, Any]
    tokenizer_path: Optional[str] = None  # Path to a directory containing a tokenizer.json file
    validation_file: Optional[str] = None
    weight_path: Optional[str] = None

@dataclass
class ModelConfig:
    architecture: str
    dimensions: Dict[str, int]
    attention: Dict[str, Any]
    normalization: Dict[str, float]
    rope: Dict[str, Any]
    misc: Dict[str, bool]

@dataclass
class TrainingConfig:
    hyperparameters: Dict[str, Any]
    scheduler: Dict[str, Any]
    optimization: Dict[str, Any]
    epochs: Optional[int] = None

@dataclass
class LoggingConfig:
    log_dir: str
    checkpoint_dir: str
    steps: Dict[str, int]
    metrics: Dict[str, bool]
    # Default to 0 (no validation) if not specified

@dataclass
class SystemConfig:
    seed: int
    device: str

@dataclass
class ResumeConfig:
    checkpoint: str  # Path to checkpoint base name
    reset_optimizer: bool = False  # Optional flag to reset optimizer state

@dataclass
class Config:
    name: str  # New field for run name
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig
    system: SystemConfig
    resume: Optional[ResumeConfig] = None
    overwrite: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Validate that name is present
        if 'name' not in config_dict:
            raise ValueError("Config must specify a 'name' field at the top level")
            
        # Extract epochs if it exists at the top level of training config
        training_config = config_dict['training'].copy()
        epochs = training_config.pop('epochs', None)
        
        # Extract resume config if present
        resume_config = None
        if 'resume' in config_dict:
            resume_config = ResumeConfig(**config_dict['resume'])
        
        return cls(
            name=config_dict['name'],
            overwrite=config_dict.get('overwrite', False),
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**training_config, epochs=epochs),
            logging=LoggingConfig(**config_dict['logging']),
            system=SystemConfig(**config_dict['system']),
            resume=resume_config
        )

class CheckpointManager:
    @staticmethod
    def validate_unique_name(name: str) -> None:
        """Validates that the run directory doesn't already exist"""
        run_path = Path('runs') / name
        if run_path.exists():
            raise ValueError(f"Run directory already exists for name '{name}'")
            
    @staticmethod
    def setup_run_directory(name: str) -> tuple[Path, Path, Path]:
        """Creates and returns paths for run directory structure"""
        run_dir = Path('runs') / name
        checkpoint_dir = run_dir / 'checkpoints'
        
        # Create directory structure
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(exist_ok=True)
        
        return run_dir, run_dir / 'log.txt', checkpoint_dir
        
    @staticmethod
    def get_checkpoint_paths(checkpoint_path: str) -> tuple[str, str, str]:
        """Returns the paths for model, optimizer, and state files"""
        model_path = f"{checkpoint_path}_model.safetensors"
        optimizer_path = f"{checkpoint_path}_optimizer.safetensors"
        state_path = f"{checkpoint_path}_state.json"
        return model_path, optimizer_path, state_path

class TokenizerManager:
    def __init__(self, config: DataConfig, run_dir: Optional[Path] = None):
        self.config = config
        self.external_tokenizer = None
        
        # Check if an external tokenizer path is provided
        if config.tokenizer_path is not None:
            self.use_external_tokenizer(config.tokenizer_path)
            
            # If we have a run directory, copy the tokenizer to it
            if run_dir is not None:
                self.copy_tokenizer_to_run_dir(config.tokenizer_path, run_dir)
        else:
            # Fall back to byte-level tokenization
            self.setup_vocabulary()
    
    def use_external_tokenizer(self, tokenizer_path: str):
        """Load and use an external tokenizer from the specified path."""
        from tokenizers import Tokenizer
        import os
        tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
        
        if not os.path.exists(tokenizer_file):
            raise ValueError(f"Tokenizer file not found at {tokenizer_file}")
        
        print(f"Loading external tokenizer from {tokenizer_file}")
        self.external_tokenizer = Tokenizer.from_file(tokenizer_file)
        
        # Extract special token IDs
        vocab = self.external_tokenizer.get_vocab()
        special_tokens = self.config.tokenizer['special_tokens']
        
        # Map special tokens to their IDs
        self.PAD_TOKEN = vocab.get(special_tokens['pad'])
        self.BOS_TOKEN = vocab.get(special_tokens['bos'])
        self.EOS_TOKEN = vocab.get(special_tokens['eos'])
        self.VOCAB_SIZE = len(vocab)
        
        if self.PAD_TOKEN is None or self.BOS_TOKEN is None or self.EOS_TOKEN is None:
            raise ValueError(f"One or more special tokens not found in the external tokenizer vocabulary")
    
    def copy_tokenizer_to_run_dir(self, tokenizer_path: str, run_dir: Path):
        """Copy the tokenizer files to the run directory."""
        import shutil
        import os
        
        # Create tokenizer directory in run_dir
        run_tokenizer_dir = run_dir / 'tokenizer'
        os.makedirs(run_tokenizer_dir, exist_ok=True)
        
        # Copy tokenizer.json
        tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
        shutil.copy2(tokenizer_file, run_tokenizer_dir / "tokenizer.json")
        
        print(f"Copied tokenizer to {run_tokenizer_dir}")
        
    def setup_vocabulary(self):
        """Set up the byte-level tokenization vocabulary."""
        normal_vocab_size = self.config.tokenizer['normal_vocab_size']
        special_tokens = self.config.tokenizer['special_tokens']
        
        # Create vocabulary mapping
        self.special_token_map = {
            token: normal_vocab_size + idx 
            for idx, token in enumerate(special_tokens.values())
        }
        
        # Store common tokens
        self.PAD_TOKEN = self.special_token_map[special_tokens['pad']]
        self.BOS_TOKEN = self.special_token_map[special_tokens['bos']]
        self.EOS_TOKEN = self.special_token_map[special_tokens['eos']]
        self.VOCAB_SIZE = normal_vocab_size + len(self.special_token_map)
        
    def tokenize(self, text: str) -> list:
        if self.external_tokenizer is not None:
            # Use external tokenizer
            encoded = self.external_tokenizer.encode(text)
            return encoded.ids
        else:
            # Use byte-level tokenization
            return list(text.encode('utf-8'))
            
    def detokenize(self, tokens: list) -> str:
        if self.external_tokenizer is not None:
            # Use external tokenizer
            return self.external_tokenizer.decode(tokens.tolist())
        else:
            # Use byte-level detokenization
            return bytes(tokens).decode('utf-8', errors='ignore')
            
    def tokenize_doc(self, doc: str) -> list:
        """Tokenize a document, ensuring it doesn't exceed the max context size.
        
        Args:
            doc: The text to tokenize
            
        Returns:
            A list of token IDs, including BOS and EOS tokens
        """
        max_length = self.config.preprocessing['max_context_size']
        
        if self.external_tokenizer is not None:
            # Use external tokenizer
            encoded = self.external_tokenizer.encode(doc)
            tokens = encoded.ids[:max_length]
            return [self.BOS_TOKEN] + tokens + [self.EOS_TOKEN]
        else:
            # Use byte-level tokenization
            return [self.BOS_TOKEN] + self.tokenize(doc)[:max_length] + [self.EOS_TOKEN]

class DataManager:
    def __init__(self, config: DataConfig, tokenizer: TokenizerManager, batch_size: int = 1):
        self.config = config
        self.tokenizer = tokenizer
        self.train_docs = []
        self.val_docs = []
        self.train_idx = None
        self.val_idx = None
        self.batch_size = batch_size
        self.load_data()
       
    def load_data(self):
        # Load training data
        self._load_file(self.config.input_file, self.train_docs)
        
        # Set up training batches
        self.train_idx = sorted(range(len(self.train_docs)), key=lambda idx: len(self.train_docs[idx]))
        random.shuffle(self.train_idx)
        self.train_batch_idx = [
            self.train_idx[i : i + self.batch_size : 1]
            for i in range(0, len(self.train_idx) - self.batch_size + 1, self.batch_size)
        ]
        self.train_indices = np.random.permutation(len(self.train_batch_idx))
        
        # Load validation data if specified
        if self.config.validation_file:
            self._load_file(self.config.validation_file, self.val_docs)
            
            # Set up validation batches
            self.val_idx = sorted(range(len(self.val_docs)), key=lambda idx: len(self.val_docs[idx]))
            self.val_batch_idx = [
                self.val_idx[i : i + self.batch_size : 1]
                for i in range(0, len(self.val_idx) - self.batch_size + 1, self.batch_size)
            ]
            self.val_indices = np.random.permutation(len(self.val_batch_idx))
            self.val_ptr = 0  # Pointer for validation batches
            
    def _load_file(self, file_path: str, docs_list: list):
        """Helper method to load documents from a file."""
        with open(file_path, 'r') as f:
            for line in f:
                d = json.loads(line)
                text = d["text"]
                """chunk_size = self.config.preprocessing['max_context_size']
                overlap = self.config.preprocessing.get('chunk_overlap', 0)
                
                # Handle overlapping chunks if specified
                stride = chunk_size - overlap
                for i in range(0, len(text), stride):
                    chunk_text = text[i : i + chunk_size]
                    docs_list.append(chunk_text)"""
                docs_list.append(text)
    
    def generate_batch(self, step: int) -> mx.array:
        """Generate a training batch."""
        indices = self.train_batch_idx[self.train_indices[step % len(self.train_indices)]]
        return self._create_batch([self.train_docs[i] for i in indices])
    
    def generate_validation_batch(self, batch_idx: int) -> mx.array:
        """Generate a validation batch."""
        if not self.config.validation_file or batch_idx >= len(self.val_batch_idx):
            raise ValueError("No validation data available or batch index out of range")
        
        indices = self.val_batch_idx[self.val_indices[self.val_ptr % len(self.val_indices)]]
        self.val_ptr += 1
        return self._create_batch([self.val_docs[i] for i in indices])
    
    def _create_batch(self, docs: list) -> mx.array:
        """Helper method to create and pad a batch from documents."""
        batch = [self.tokenizer.tokenize_doc(doc) for doc in docs]
        max_len = max(len(x) for x in batch)
        
        # Pad sequences
        for i in range(len(batch)):
            batch[i] += [self.tokenizer.PAD_TOKEN] * (max_len - len(batch[i]))
            
        return mx.array(batch)
    
    @property
    def has_validation_data(self) -> bool:
        """Check if validation data is available."""
        return self.config.validation_file is not None and len(self.val_docs) > 0
    
    @property
    def num_validation_batches(self) -> int:
        """Get the number of validation batches."""
        return len(self.val_batch_idx) if self.has_validation_data else 0

class OptimizationManager:
    def __init__(self, config: TrainingConfig, num_training_steps: int):
        self.config = config
        self.num_training_steps = num_training_steps
        
    def create_scheduler(self) -> Any:
        cfg = self.config.scheduler
        initial_lr = self.config.hyperparameters['learning_rate']
        
        if cfg['type'] == 'cosine_with_warmup':
            warmup = optim.linear_schedule(0, initial_lr, steps=cfg['warmup_steps'])
            cosine = optim.cosine_decay(initial_lr, self.num_training_steps, initial_lr * cfg['min_lr_ratio'])
            return optim.join_schedules([warmup, cosine], [cfg['warmup_steps']])
        elif cfg['type'] == 'cosine':
            return optim.cosine_decay(initial_lr, self.num_training_steps, initial_lr * cfg['min_lr_ratio'])
        elif cfg['type'] == 'linear':
            return optim.linear_schedule(initial_lr, 0, steps=self.num_training_steps)
        else:
            raise ValueError(f"Unsupported scheduler type: {cfg['type']}")
            
    def create_optimizer(self, schedule: Any) -> optim.Optimizer:
        cfg = self.config.optimization
        kwargs = {
            'learning_rate': schedule,
        }
        if 'betas' in cfg:
            kwargs['betas'] = tuple(cfg['betas'])
        if 'eps' in cfg:
            kwargs['eps'] = cfg['eps']
        if 'weight_decay' in cfg:
            kwargs['weight_decay'] = self.config.hyperparameters['weight_decay']
        if cfg['optimizer'] == 'adamw':
            return optim.AdamW(**kwargs)
        elif cfg['optimizer'] == 'adam':
            return optim.Adam(**kwargs)
        elif cfg['optimizer'] == 'muon':
            return optim_x.Muon(**kwargs, alternate_optimizer=optim.AdamW(**kwargs))
        elif cfg['optimizer'] == 'sgd':
            return optim.SGD(**kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {cfg['optimizer']}")

class Trainer:
    def __init__(self, config_path: str, for_training=True):
        self.config = Config.from_yaml(config_path)
        self.config_path = config_path
        
        # Initialize tracking variables
        self.total_tokens = 0
        self.start_step = 0
        
        # Validate unique run name before proceeding
        if for_training and not self.config.overwrite and not (self.config.resume and self.config.resume.checkpoint):
            CheckpointManager.validate_unique_name(self.config.name)
        
        self.setup_system()
        
        # Create run directory early so we can copy tokenizer to it
        if for_training:
            self.run_dir, self.log_file, self.checkpoint_dir = CheckpointManager.setup_run_directory(self.config.name)
        else:
            self.run_dir = None
            
        # Initialize tokenizer with run directory if available
        self.tokenizer = TokenizerManager(self.config.data, self.run_dir)
        
        self.setup_model()
        if for_training:
            self.data_manager = DataManager(self.config.data, self.tokenizer, batch_size=self.config.training.hyperparameters['batch_size'])
            self.setup_training()
            self.setup_logging()
            
            # Initialize validation metrics tracking
            self.validation_steps = self.config.logging.steps.get('validation_interval', 0)
            self.validation_losses = []
    
    def setup_system(self):
        # Set random seeds
        random.seed(self.config.system.seed)
        np.random.seed(self.config.system.seed)
        mx.random.seed(self.config.system.seed)
        
    def setup_model(self):
        model_cfg = self.config.model
        arch_file = f"arch.{model_cfg.architecture}"
        mlx_lm_file = f"mlx_lm.models.{model_cfg.architecture}"
        Model = None
        ModelArgs = None
        try:
            module = importlib.import_module(arch_file)
            Model = getattr(module, 'Model')
            ModelArgs = getattr(module, 'ModelArgs')
        except ImportError:
            try:
                module = importlib.import_module(mlx_lm_file)
                Model = getattr(module, 'Model')
                ModelArgs = getattr(module, 'ModelArgs')
            except ImportError:
                raise ImportError(f"Model architecture '{model_cfg.architecture}' not found in both {arch_file} and {mlx_lm_file}")
        
        all_args = {
            'model_type': model_cfg.architecture,
            'hidden_size': model_cfg.dimensions['hidden_size'],
            'num_hidden_layers': model_cfg.dimensions.get('num_layers', 8),
            'intermediate_size': model_cfg.dimensions['intermediate_size'],
            'num_attention_heads': model_cfg.attention['num_heads'],
            'rms_norm_eps': model_cfg.normalization['rms_norm_eps'],
            'vocab_size': self.tokenizer.VOCAB_SIZE,
            'head_dim': model_cfg.attention['head_dim'],
            'max_position_embeddings': model_cfg.attention['max_position_embeddings'],
            'num_key_value_heads': model_cfg.attention['num_kv_heads'],
            'attention_bias': model_cfg.misc['attention_bias'],
            'mlp_bias': model_cfg.misc['mlp_bias'],
            'rope_theta': model_cfg.rope['theta'],
            'rope_traditional': model_cfg.rope['traditional'],
            'rope_scaling': model_cfg.rope['scaling'],
            'tie_word_embeddings': model_cfg.misc['tie_word_embeddings'],
            'logit_scale': model_cfg.misc.get('logit_scale', None),
            'num_local_experts': model_cfg.misc.get('num_local_experts', 0),
            'num_experts_per_tok': model_cfg.misc.get('num_experts_per_tok', 0),
        }
        valid_args = filter_valid_args(ModelArgs, all_args)
        args = ModelArgs(**valid_args)

        self.model = Model(args)

        if self.config.data.weight_path is not None:
            print(f"Loading weights from {self.config.data.weight_path}")
            self.model.load_weights(self.config.data.weight_path, strict=False)
        # Log model size
        p = sum(v.size for _, v in tree_flatten(self.model.trainable_parameters())) / 10**6
        print(f"Model has {p:.2f}M parameters")
        
    def setup_training(self):
        # Calculate number of training steps
        num_samples = len(self.data_manager.train_docs)
        batch_size = self.config.training.hyperparameters['batch_size']
        steps_per_epoch = num_samples // batch_size
        
        if self.config.training.epochs is not None:
            # If epochs is set, calculate total steps based on epochs
            self.total_steps = steps_per_epoch * self.config.training.epochs
        else:
            # Otherwise use specified iters or default to one epoch
            self.total_steps = self.config.training.hyperparameters.get('iters', steps_per_epoch)
        
        # Store steps_per_epoch for logging
        self.steps_per_epoch = steps_per_epoch
        
        # Setup optimization
        opt_manager = OptimizationManager(self.config.training, self.total_steps)
        self.lr_schedule = opt_manager.create_scheduler()
        self.optimizer = opt_manager.create_optimizer(self.lr_schedule)
        
    def setup_logging(self):
        # Run directory structure should already be set up in __init__
        
        # Create initial metadata file
        metadata = {
            'name': self.config.name,
            'created_at': datetime.now().isoformat(),
            'config': {
                'model': self.config.model.__dict__,
                'training': self.config.training.__dict__,
                'system': self.config.system.__dict__
            },
            'training_info': {
                'steps_per_epoch': self.steps_per_epoch,
                'total_steps': self.total_steps,
                'epochs': self.config.training.epochs
            }
        }
        
        # Add tokenizer information to metadata
        if self.config.data.tokenizer_path:
            metadata['tokenizer'] = {
                'type': 'external',
                'path': self.config.data.tokenizer_path,
                'vocab_size': self.tokenizer.VOCAB_SIZE
            }
        else:
            metadata['tokenizer'] = {
                'type': 'byte-level',
                'vocab_size': self.tokenizer.VOCAB_SIZE
            }
        
        with open(self.run_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Save the config used to the run directory
        with open(self.run_dir / 'config.yaml', 'w') as f:
            with open(self.config_path, 'r') as config_file:
                f.write(config_file.read())
    
    def compute_loss(self, model, inputs: mx.array, targets: mx.array) -> Tuple[mx.array, int]:
        logits = model(inputs)
        logits = logits.astype(mx.float32)
        loss = nn.losses.cross_entropy(logits, targets)
        # Mask padding tokens
        pad_mask = (targets != self.tokenizer.PAD_TOKEN)
        loss = loss * pad_mask
        ntoks = pad_mask.sum()
        
        return loss.sum() / ntoks, ntoks
        
    def validate(self) -> float:
        """Run validation on the validation dataset.
        
        Returns:
            float: Average validation loss
        """
        if not self.data_manager.has_validation_data:
            return None
            
        # Ensure we're in evaluation mode (no need for gradients)
        total_loss = 0.0
        total_tokens = 0
        
        # Process all validation batches
        num_batches = min(self.data_manager.num_validation_batches, 50)  # Cap at 50 batches to avoid too long validation
        
        for batch_idx in range(num_batches):
            batch = self.data_manager.generate_validation_batch(batch_idx)
            
            # Forward pass only
            loss, tokens = self.compute_loss(self.model, batch[:, :-1], batch[:, 1:])
            
            # Accumulate metrics
            total_loss += float(loss)
            total_tokens += tokens
            
            # Clear GPU cache if needed
            if self.config.system.device == "gpu":
                mx.clear_cache()
        
        # Calculate average loss
        avg_loss = total_loss / num_batches
        
        return avg_loss

    def save_checkpoint(self, step: int | str, val_loss: float = None):
        # Save model weights
        weights = dict(tree_flatten(self.model.parameters()))
        model_path = self.checkpoint_dir / f'step_{step}_model.safetensors'
        mx.save_safetensors(str(model_path), weights)
        
        # Save optimizer state
        optimizer_state = dict(tree_flatten(self.optimizer.state))
        optimizer_path = self.checkpoint_dir / f'step_{step}_optimizer.safetensors'
        mx.save_safetensors(str(optimizer_path), optimizer_state)
        
        # Save training state
        training_state = {
            'step': step if isinstance(step, int) else self.total_steps,
            'val_ptr': self.data_manager.val_ptr,
            'total_tokens': self.total_tokens.item(),
            'validation_losses': self.validation_losses,
        }
        state_path = self.checkpoint_dir / f'step_{step}_state.json'
        with open(state_path, 'w') as f:
            json.dump(training_state, f)
        
        # Update metadata with checkpoint info
        metadata_path = self.run_dir / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        if 'checkpoints' not in metadata:
            metadata['checkpoints'] = []
        
        checkpoint_info = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'paths': {
                'model': f'checkpoints/step_{step}_model.safetensors',
                'optimizer': f'checkpoints/step_{step}_optimizer.safetensors',
                'state': f'checkpoints/step_{step}_state.json'
            }
        }
        
        # Include validation loss if available
        if val_loss is not None:
            checkpoint_info['validation_loss'] = val_loss
            
        metadata['checkpoints'].append(checkpoint_info)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def log_metrics(self, step: int, loss: float, tokens: int, 
                   total_tokens: int, start_time: float, val_loss: float = None) -> str:
        metrics = []
        
        # Add epoch information if epochs are configured
        if self.config.training.epochs is not None:
            current_epoch = step // self.steps_per_epoch + 1
            epoch_step = step % self.steps_per_epoch + 1
            metrics.append(f"epoch={current_epoch}/{self.config.training.epochs} ({epoch_step}/{self.steps_per_epoch})")
        
        if self.config.logging.metrics['log_loss']:
            metrics.append(f"loss={loss:.3e}")
            
            # Add validation loss if available
            if val_loss is not None:
                metrics.append(f"val_loss={val_loss:.3e}")
            
        if self.config.logging.metrics['log_perplexity']:
            metrics.append(f"ppl={np.exp(loss):.2f}")
            
            # Add validation perplexity if available
            if val_loss is not None:
                metrics.append(f"val_ppl={np.exp(val_loss):.2f}")
            
        if self.config.logging.metrics['log_tokens_per_second']:
            tokens_per_sec = total_tokens / (1000 * (time.time() - start_time))
            metrics.append(f"tok/s={tokens_per_sec:.2f}K")
        
        if self.config.logging.metrics['log_tokens_processed']:
            metrics.append(f"toks={tokens}")
            
        if self.config.logging.metrics['log_learning_rate']:
            metrics.append(f"lr={self.lr_schedule(step):.3e}")
            
        return " | ".join(metrics)

    def load_checkpoint(self, checkpoint_path: str, reset_optimizer: bool = False):
        """Load a checkpoint and restore model, optimizer, and training state"""
        # Extract step from checkpoint path
        step_str = checkpoint_path.split('step_')[-1]
        
        # Get checkpoint file paths
        model_path, optimizer_path, state_path = CheckpointManager.get_checkpoint_paths(checkpoint_path)
        
        # Load model weights
        print(f"Loading model weights from {model_path}")
        #weights = mx.load(model_path)
        self.model.load_weights(model_path)
        # Load optimizer state if not resetting
        if not reset_optimizer:
            print(f"Loading optimizer state from {optimizer_path}")
            state_dict = mx.load(optimizer_path)
            state = tree_unflatten(list(state_dict.items()))
            self.optimizer.state = state
        
        # Load training state
        print(f"Loading training state from {state_path}")
        with open(state_path, 'r') as f:
            training_state = json.load(f)
        
        # Restore training state
        self.start_step = training_state['step'] if isinstance(training_state['step'], int) else 0
        self.data_manager.val_ptr = training_state['val_ptr']
        self.total_tokens = training_state['total_tokens']
        self.validation_losses = training_state['validation_losses']
        
        print(f"Resumed training from checkpoint {checkpoint_path} at step {self.start_step}")
        
        return self.start_step

    def train(self):
        # Initialize variables
        total_tokens = self.total_tokens
        start_step = 0
        
        # Check if resuming from checkpoint
        if self.config.resume and self.config.resume.checkpoint:
            checkpoint_path = self.config.resume.checkpoint
            reset_optimizer = self.config.resume.reset_optimizer
            start_step = self.load_checkpoint(checkpoint_path, reset_optimizer)
            
            # If we're resuming, we should skip the initial validation
            skip_initial_validation = True
        else:
            skip_initial_validation = False
            
        loss_value_and_grad = nn.value_and_grad(self.model, self.compute_loss)
        start_time = time.time()
        # Create progress bar with adjusted range for resuming
        progress_bar = tqdm(range(self.total_steps), desc="Training", initial=start_step)

        
        # Initialize logging
        with open(self.log_file, 'a' if start_step > 0 else 'w') as log_file:
            if start_step == 0:
                log_file.write(f"Training started at {datetime.now()}\n")
                log_file.write(f"Total steps: {self.total_steps}\n")
                if self.config.training.epochs is not None:
                    log_file.write(f"Training for {self.config.training.epochs} epochs with {self.steps_per_epoch} steps per epoch\n")
                if self.data_manager.has_validation_data:
                    log_file.write(f"Validation data: {self.config.data.validation_file}\n")
                    log_file.write(f"Validation batches: {self.data_manager.num_validation_batches}\n")
                log_file.write("=" * 50 + "\n\n")
            else:
                log_file.write(f"\nResuming training at step {start_step} at {datetime.now()}\n")
                log_file.write(f"Remaining steps: {self.total_steps - start_step}\n")
                log_file.write("=" * 50 + "\n\n")
            
            # Log initial validation loss if validation data is available and not resuming
            val_loss = None
            if self.validation_steps > 0 and self.data_manager.has_validation_data and not skip_initial_validation:
                val_loss = self.validate()
                log_file.write(f"Initial validation loss: {val_loss:.4e} (ppl={np.exp(val_loss):.2f})\n\n")
                # Add to validation loss history
                self.validation_losses.append((0, val_loss))
            
            for step in progress_bar:
                step += start_step
                if step >= self.total_steps:
                    break
                # Generate batch
                batch = self.data_manager.generate_batch(step)
                
                # Forward and backward pass
                (loss, tokens), grad = loss_value_and_grad(
                    self.model, batch[:, :-1], batch[:, 1:]
                )
                
                # Gradient clipping if configured
                if 'gradient_clip' in self.config.training.hyperparameters:
                    clip_value = self.config.training.hyperparameters['gradient_clip']
                    grad = tree_map(lambda x: mx.clip(x, -clip_value, clip_value), grad)
                
                # Update model
                total_tokens += tokens
                self.optimizer.update(self.model, grad)
                mx.eval(loss)
                
                if self.config.system.device == "gpu":
                    mx.clear_cache()
                
                # Run validation
                if self.validation_steps > 0 and self.data_manager.has_validation_data and (step + 1) % self.validation_steps == 0:
                    val_loss = self.validate()
                    # Add to validation loss history
                    self.validation_losses.append((step + 1, val_loss))
                    
                    # Log validation separately for clear visibility
                    val_metrics = f"val_loss={val_loss:.3e} | val_ppl={np.exp(val_loss):.2f}"
                    log_file.write(f"Step {step + 1} validation: {val_metrics}\n")
                    log_file.flush()
                
                # Logging
                if step % self.config.logging.steps['logging_interval'] == 0:
                    # Only include val_loss if it was just calculated
                    current_val_loss = val_loss if self.validation_steps > 0 and (step + 1) % self.validation_steps == 0 else None
                    metrics = self.log_metrics(step, loss, tokens, total_tokens, start_time, current_val_loss)
                    
                    # Update progress bar
                    progress_bar.set_description(metrics)
                    
                    # Write to log file
                    log_message = f"Step {step}: {metrics}\n"
                    log_file.write(log_message)
                    log_file.flush()
                
                # Save checkpoint
                if (1 + step) % self.config.logging.steps['checkpoint_interval'] == 0:
                    # Find the most recent validation loss if available
                    last_val_loss = val_loss if val_loss is not None else None
                    # Update total_tokens in the trainer instance for checkpoint saving
                    self.total_tokens = total_tokens
                    self.save_checkpoint(step + 1, last_val_loss)
        
        # Final validation
        final_val_loss = None
        if self.validation_steps > 0 and self.data_manager.has_validation_data:
            final_val_loss = self.validate()
            self.validation_losses.append((self.total_steps, final_val_loss))
        
        # Save final checkpoint with validation loss
        self.total_tokens = total_tokens
        self.save_checkpoint("final", final_val_loss)
        
        # Save validation losses to metadata
        if self.validation_losses:
            metadata_path = self.run_dir / 'metadata.json'
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            metadata['validation'] = {
                'steps': [step for step, _ in self.validation_losses],
                'losses': [float(loss) for _, loss in self.validation_losses]
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Write final summary
        with open(self.log_file, 'a') as log_file:
            log_file.write("\n" + "=" * 50 + "\n")
            log_file.write(f"Training completed at {datetime.now()}\n")
            log_file.write(f"Final training metrics: {metrics}\n")
            if final_val_loss is not None:
                log_file.write(f"Final validation loss: {final_val_loss:.4e} (ppl={np.exp(final_val_loss):.2f})\n")
            log_file.write(f"Total tokens processed: {total_tokens/1000:.2f}K\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train a language model with MLX')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    args = parser.parse_args()
    # Make 'runs' directory if it doesn't exist
    os.makedirs('runs', exist_ok=True)
    trainer = Trainer(args.config)
    trainer.train()

if __name__ == "__main__":
    main()
