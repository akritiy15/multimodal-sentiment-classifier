import torch
import torch.nn as nn
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType

class MultimodalSentimentModel(nn.Module):
    def __init__(
        self,
        model_name="bert-base-uncased",
        num_labels=6,
        structured_feat_dim=10,
        dropout=0.3,
        use_lora=True
    ):
        super().__init__()

        # Load pre-trained BERT backbone
        backbone = AutoModel.from_pretrained(model_name)

        # Wrap with LoRA if requested
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=8,                    # LoRA rank — lower = less memory
                lora_alpha=16,          # scaling factor
                lora_dropout=0.1,
                target_modules=["query", "value"]  # only adapt attention
            )
            self.bert = get_peft_model(backbone, lora_config)
            self.bert.print_trainable_parameters()
        else:
            self.bert = backbone

        # BERT hidden size (768 for bert-base)
        bert_hidden = self.bert.config.hidden_size

        # Fusion + classification head
        # Input = BERT [CLS] embedding (768) + structured features (10)
        fused_dim = bert_hidden + structured_feat_dim

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, structured_features,
                token_type_ids=None):
        # Pass text through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # [CLS] token embedding — shape: (batch, 768)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Concatenate text embedding + structured features
        # shape: (batch, 768 + 10) = (batch, 778)
        fused = torch.cat([cls_embedding, structured_features], dim=1)

        # Final classification
        logits = self.classifier(fused)
        return logits