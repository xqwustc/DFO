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

import os
import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, LogisticRegression, MaskedFeatureEmbedding
import numpy as np
from tqdm import tqdm
import sys
import logging
from torch import log
import pandas as pd
from tqdm import tqdm
from feat_select.AdaFS_module import AdaFS
import sys
from model_zoo.utils import get_gates_prob_dfo,get_sum_feature_dimensions,get_gates_prob_autofield,permute_feature
from itertools import cycle
import torch.optim as optim


EPS = 1e-6
lamda = 1e-5 # 1e-3
lamda_opt = 2e-9
class WideDeep(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="WideDeep", 
                 gpu=-1, 
                 learning_rate=1e-3,
                 select_num=0,
                 embedding_dim=10, 
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(WideDeep, self).__init__(feature_map, 
                                       model_id=model_id, 
                                       gpu=gpu, 
                                       embedding_regularizer=embedding_regularizer, 
                                       net_regularizer=net_regularizer,
                                       **kwargs)

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        self.learning_rate = learning_rate

        self.lr_layer = LogisticRegression(feature_map, use_bias=False)
        self.dnn = MLP_Block(input_dim=get_sum_feature_dimensions(self.embedding_layer),
                             #embedding_dim*len(feature_map.features),#input_dim=embedding_dim * feature_map.num_fields,
                             output_dim=1,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)

        self.lr_embedding = FeatureEmbedding(self.feature_map, 1, use_pretrain=False, use_sharing=False)


        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        # feature_emb = self.embedding_layer(X)
        feature_emb = self.embedding_layer(X,flatten_emb=True)
        y_pred = self.lr_layer(X)
        # y_pred += self.dnn(feature_emb.flatten(start_dim=1))
        y_pred += self.dnn(feature_emb)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def forward_with_dfo(self, inputs, gates_prob):
        X = self.get_inputs(inputs)

        feature_emb = self.embedding_layer(X)
        emb_weights = self.lr_embedding(X)

        for i in range(len(self.feature_map.features)):
            feature_emb[:,i:i+1,:] *= gates_prob[i]
            emb_weights[:,i:i+1,:] *= gates_prob[i]


        y_pred = emb_weights.sum(dim=1)
        y_pred += self.dnn(feature_emb.flatten(start_dim=1))
        y_pred = self.output_activation(y_pred)
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

        torch.autograd.set_detect_anomaly(True)
        # Make a list of theta for each feature
        gates_theta = torch.ones(len(self.feature_map.features)) * 0.5
        gates_theta.requires_grad_(requires_grad=True)

        gates_sigma = torch.ones(len(self.feature_map.features)) * 0.5
        gates_sigma.requires_grad_(requires_grad=True)

        self.optimizer.add_param_group({'params': gates_theta, 'lr': self.learning_rate * 0.1})
        self.optimizer.add_param_group({'params': gates_sigma, 'lr': self.learning_rate * 0.1})

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
                gates_prob = get_gates_prob_dfo(gates_theta, gates_sigma, tau=kwargs.get("tau", None))
                return_dict = self.forward_with_dfo(batch_data, gates_prob)

                self.optimizer.zero_grad()

                y_true = self.get_labels(batch_data)
                loss = self.compute_loss(return_dict, y_true)

                loss += torch.sum(gates_prob) * 1e-5


                loss.backward()

                nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
                self.optimizer.step()

                train_loss += loss.item()
                if self._total_steps % self._eval_steps == 0:
                    logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                    train_loss = 0
                    # eval the model
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