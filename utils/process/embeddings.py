import numpy as np
import torch
from torch_geometric.data import Data
from utils.functions import tokenizer
from utils.functions import log as logger
from gensim.models.keyedvectors import Word2VecKeyedVectors
from models.layers import encode_input
from transformers import RobertaTokenizer, RobertaModel

class NodesEmbedding:
    def __init__(self, nodes_dim: int, w2v_keyed_vectors: Word2VecKeyedVectors):
        self.w2v_keyed_vectors = w2v_keyed_vectors
        self.kv_size = w2v_keyed_vectors.vector_size
        self.tokenizer_bert = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.bert_model = RobertaModel.from_pretrained("microsoft/codebert-base").to("cuda")
        self.bert_model.eval()
        self.nodes_dim = nodes_dim
        self.bert_hidden_size = 768 

        assert self.nodes_dim >= 0

    def __call__(self, nodes):
        embedded_nodes = self.embed_nodes(nodes)
        return torch.from_numpy(embedded_nodes).float()

    def embed_nodes(self, nodes):
        embeddings = []
        
        valid_nodes_data = []
        
        for n_id, node in nodes.items():
            node_code = node.get_code()
            tokenized_code = tokenizer(node_code, True)
            valid_nodes_data.append((n_id, node, tokenized_code))
        
        all_input_ids = []
        all_attention_masks = []
        node_types = []
        for n_id, node, tokenized_code in valid_nodes_data:
            str_tokens = ' '.join(tokenized_code)
            input_ids, attention_mask = encode_input(str_tokens, self.tokenizer_bert)
            
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)  
            node_types.append(node.type)
        
        # Batch các tensors lại
        batch_input_ids = torch.cat(all_input_ids, dim=0)  # [batch_size, seq_len]
        batch_attention_masks = torch.cat(all_attention_masks, dim=0)
        
        # Forward pass qua BERT model với memory optimization
        with torch.no_grad():  # Không cần gradient cho embedding
            cls_feats = self.bert_model(
                batch_input_ids.to("cuda"), 
                batch_attention_masks.to("cuda")
            )[0][:, 0]  # [batch_size, hidden_size]
            
            source_embeddings = cls_feats.cpu().numpy()
        
        # Combine với node types (thứ tự đúng vì cùng iterate qua valid_nodes_data)
        for i, node_type in enumerate(node_types):
            embedding = np.concatenate((np.array([node_type]), source_embeddings[i]), axis=0)
            embeddings.append(embedding)
        
        # Aggressive cleanup
        del batch_input_ids, batch_attention_masks, cls_feats, source_embeddings
        torch.cuda.empty_cache()
        
        return np.array(embeddings)


    # fromTokenToVectors
    # This is the original Word2Vec model usage.
    # Although we keep this part of the code, we are not using it.
    def get_vectors(self, tokenized_code, node):
        vectors = []

        for token in tokenized_code:
            if token in self.w2v_keyed_vectors.vocab:
                vectors.append(self.w2v_keyed_vectors[token])
            else:
                # print(node.label, token, node.get_code(), tokenized_code)
                vectors.append(np.zeros(self.kv_size))
                if node.label not in ["Identifier", "Literal", "MethodParameterIn", "MethodParameterOut"]:
                    msg = f"No vector for TOKEN {token} in {node.get_code()}."
                    logger.log_warning('embeddings', msg)

        return vectors


class GraphsEmbedding:
    def __init__(self, edge_type):
        self.edge_types = [et.strip() for et in edge_type.split(',')]

    def __call__(self, nodes):
        connections = self.nodes_connectivity(nodes)

        return torch.tensor(connections).long()

    # nodesToGraphConnectivity
    def nodes_connectivity(self, nodes):
        # nodes are ordered by line and column
        coo = [[], []]

        for node_idx, (node_id, node) in enumerate(nodes.items()):
            if node_idx != node.order:
                raise Exception("Something wrong with the order")

            for e_id, edge in node.edges.items():
                if edge.type not in self.edge_types:
                    print("Skip edge type: ", edge.type)
                    continue

                if edge.node_in in nodes and edge.node_in != node_id:
                    coo[0].append(nodes[edge.node_in].order)
                    coo[1].append(node_idx)

                if edge.node_out in nodes and edge.node_out != node_id:
                    coo[0].append(node_idx)
                    coo[1].append(nodes[edge.node_out].order)
                    
        return coo


# Global BERT model instance để tránh memory leak
_global_embedding = None

def nodes_to_input(nodes, target, nodes_dim, keyed_vectors, edge_type):
    global _global_embedding
    
    if _global_embedding is None or _global_embedding.nodes_dim != nodes_dim:
        _global_embedding = NodesEmbedding(nodes_dim, keyed_vectors)
    
    graphs_embedding = GraphsEmbedding(edge_type)
    label = torch.tensor([target]).float()

    return Data(x=_global_embedding(nodes), edge_index=graphs_embedding(nodes), y=label)
