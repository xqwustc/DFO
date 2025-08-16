# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
import numpy as np
from itertools import cycle
from model_zoo.utils import get_gates_prob_dfo
from torch import optim
import logging
import sys
import pandas as pd
from tqdm import tqdm
from itertools import cycle

class DNN(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="DNN",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 hidden_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(DNN, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.learning_rate = learning_rate
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.mlp = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=self.output_activation,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)


        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def get_total_parameters(self):
        # only embedding layer and controller params will be counted
        params = 0

        # embedding layer
        for emb in  self.embedding_layer.embedding_layer.embedding_layers.values():
            params += sum(p.numel() for p in emb.parameters())

        # controller
        if hasattr(self,'controller'):
            for param in self.controller.parameters():
                params += param.numel()

        return params

    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, flatten_emb=True)
        y_pred = self.mlp(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def forward_with_dfo(self, inputs, gates_prob):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)

        for i in range(len(self.feature_map.features)):
            feature_emb[:,i:i+1,:] *= gates_prob[i]

        feature_emb = feature_emb.flatten(start_dim=1)

        y_pred = self.mlp(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict


    def fit_for_dfo(self, data_generator, epochs=1, validation_data=None,
                    max_gradient_norm=10., **kwargs):
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch

        gates_theta = torch.ones(len(self.feature_map.features)) * 0.5
        gates_theta.requires_grad_(requires_grad=True)

        gates_sigma = torch.ones(len(self.feature_map.features)) * 0.5
        gates_sigma.requires_grad_(requires_grad=True)


        self.optimizer.add_param_group({'params': gates_theta, 'lr': self.learning_rate*0.1})
        self.optimizer.add_param_group({'params': gates_sigma, 'lr': self.learning_rate*0.1})

        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            self._epoch_index = epoch
            self._batch_index = 0
            train_loss = 0
            self.train()
            if self._verbose == 0:
                batch_iterator = data_generator
            else:
                batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_index, batch_data in enumerate(batch_iterator):
                self._batch_index = batch_index
                self._total_steps += 1

                # Use the important learner
                gates_prob = get_gates_prob_dfo(gates_theta, gates_sigma, tau = kwargs.get("tau", None))
                return_dict = self.forward_with_dfo(batch_data, gates_prob)

                self.optimizer.zero_grad()

                y_true = self.get_labels(batch_data)
                loss = self.compute_loss(return_dict, y_true)

                loss += (torch.sum(gates_prob)) * 1e-5

                loss.backward()

                nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
                self.optimizer.step()

                train_loss += loss.item()
                if self._total_steps % self._eval_steps == 0:
                    logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                    train_loss = 0
                    self.eval_step()
                if self._stop_training:
                    break

            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)

        # Create a DataFrame with feature names, feature weights, and feature sigma values
        feature_importance_result = pd.DataFrame({'feature_name': list(self.feature_map.features.keys()),
                                                  'feature_weight': gates_theta.tolist(),
                                                  'feature_sigma': gates_sigma.tolist()})

        # Sort the DataFrame by 'feature_weight' column in descending order
        feature_importance_result = feature_importance_result.sort_values(by='feature_weight', ascending=False)

        # Save the sorted result to a CSV file without the index column
        feature_importance_result.to_csv('feature_importance_result.csv', index=False)