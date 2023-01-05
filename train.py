#!/usr/bin/env python3
"""
NLLB (No Language Left Behind) Training Script from Scratch
Implements training for multilingual neural machine translation model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    M2M100Tokenizer, 
    M2M100ForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
import logging
from tqdm import tqdm
import json
import argparse
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLLBDataset(Dataset):
    """Custom dataset for NLLB training with multilingual support"""
    
    def __init__(self, data_path: str, tokenizer, src_lang: str, tgt_lang: str, max_length: int = 512):
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        
        # Load dataset - using OPUS-100 as example multilingual dataset
        if data_path:
            self.dataset = load_dataset('text', data_files=data_path)['train']
        else:
            # Use OPUS-100 dataset for multilingual training
            self.dataset = load_dataset('opus100', f'{src_lang}-{tgt_lang}', split='train[:10000]')  # Subset for demo
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        if 'translation' in item:
            # OPUS-100 format
            src_text = item['translation'][self.src_lang]
            tgt_text = item['translation'][self.tgt_lang]
        else:
            # Custom format - assume tab-separated
            parts = item['text'].split('\t')
            src_text = parts[0] if len(parts) > 0 else ""
            tgt_text = parts[1] if len(parts) > 1 else ""
        
        # Set source and target language tokens
        self.tokenizer.src_lang = self.src_lang
        self.tokenizer.tgt_lang = self.tgt_lang
        
        # Tokenize source
        src_inputs = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        with self.tokenizer.as_target_tokenizer():
            tgt_inputs = self.tokenizer(
                tgt_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        return {
            'input_ids': src_inputs['input_ids'].squeeze(),
            'attention_mask': src_inputs['attention_mask'].squeeze(),
            'labels': tgt_inputs['input_ids'].squeeze(),
        }

class NLLBTrainer:
    """NLLB Training Manager"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = M2M100Tokenizer.from_pretrained(config['model_name'])
        self.model = M2M100ForConditionalGeneration.from_pretrained(config['model_name'])
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = None
        self.global_step = 0
        self.best_loss = float('inf')
        
    def prepare_datasets(self):
        """Prepare training and validation datasets"""
        train_dataset = NLLBDataset(
            self.config['train_data_path'],
            self.tokenizer,
            self.config['src_lang'],
            self.config['tgt_lang'],
            self.config['max_length']
        )
        
        val_dataset = None
        if self.config.get('val_data_path'):
            val_dataset = NLLBDataset(
                self.config['val_data_path'],
                self.tokenizer,
                self.config['src_lang'],
                self.config['tgt_lang'],
                self.config['max_length']
            )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 0)
        )
        
        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config.get('num_workers', 0)
            )
        
        # Setup scheduler
        total_steps = len(self.train_loader) * self.config['num_epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
            
            # Log every N steps
            if self.global_step % self.config['log_steps'] == 0:
                logger.info(f'Step {self.global_step}, Loss: {loss.item():.4f}')
                
            # Save checkpoint every N steps
            if self.global_step % self.config['save_steps'] == 0:
                self.save_checkpoint(f'checkpoint-step-{self.global_step}')
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> float:
        """Validate the model"""
        if not self.val_loader:
            return float('inf')
            
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        logger.info(f'Validation Loss: {avg_loss:.4f}')
        return avg_loss
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.config['output_dir'], checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        torch.save({
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }, os.path.join(checkpoint_dir, 'training_state.pt'))
        
        logger.info(f'Checkpoint saved: {checkpoint_dir}')
    
    def train(self):
        """Main training loop"""
        logger.info("Starting NLLB training...")
        
        # Prepare datasets
        self.prepare_datasets()
        
        # Training loop
        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss = self.train_epoch(epoch)
            logger.info(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f}')
            
            # Validate
            val_loss = self.validate()
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint('best_model')
                logger.info(f'New best model saved with validation loss: {val_loss:.4f}')
            
            # Save epoch checkpoint
            self.save_checkpoint(f'epoch-{epoch+1}')
        
        logger.info("Training completed!")

def main():
    parser = argparse.ArgumentParser(description='Train NLLB model from scratch')
    parser.add_argument('--config', type=str, default='config.json', help='Config file path')
    parser.add_argument('--model_name', type=str, default='facebook/m2m100_418M', help='Base model name')
    parser.add_argument('--src_lang', type=str, default='en', help='Source language code')
    parser.add_argument('--tgt_lang', type=str, default='fr', help='Target language code')
    parser.add_argument('--train_data', type=str, help='Training data path')
    parser.add_argument('--val_data', type=str, help='Validation data path')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    
    args = parser.parse_args()
    
    # Load config if exists
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override with command line args
    config.update({
        'model_name': args.model_name,
        'src_lang': args.src_lang,
        'tgt_lang': args.tgt_lang,
        'train_data_path': args.train_data,
        'val_data_path': args.val_data,
        'output_dir': args.output_dir,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_length': args.max_length,
        'weight_decay': config.get('weight_decay', 0.01),
        'max_grad_norm': config.get('max_grad_norm', 1.0),
        'log_steps': config.get('log_steps', 100),
        'save_steps': config.get('save_steps', 1000),
        'num_workers': config.get('num_workers', 0)
    })
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Save config
    with open(os.path.join(config['output_dir'], 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize trainer and start training
    trainer = NLLBTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()