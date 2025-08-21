#!/usr/bin/env python3
import pandas as pd
import pickle
import torch
import numpy as np
import os
from collections import defaultdict

def check_new_embedding_data():
    """Kiểm tra dữ liệu embedding mới sau khi fix edge types"""
    print("🔍 CHECKING NEW EMBEDDING DATA")
    print("=" * 80)
    
    # Kiểm tra file input mới nhất
    input_dir = "data/input/"
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    input_files = [f for f in os.listdir(input_dir) if f.endswith('_input.pkl')]
    if not input_files:
        print(f"❌ No input files found in {input_dir}")
        return
    
    # Sắp xếp theo thời gian modify (file mới nhất)
    input_files_with_time = []
    for f in input_files:
        filepath = os.path.join(input_dir, f)
        mtime = os.path.getmtime(filepath)
        input_files_with_time.append((f, mtime))
    
    input_files_with_time.sort(key=lambda x: x[1], reverse=True)
    latest_file = input_files_with_time[0][0]
    
    print(f"📁 Latest input file: {latest_file}")
    print(f"📅 Modified time: {pd.to_datetime(input_files_with_time[0][1], unit='s')}")
    
    try:
        # Load data
        filepath = os.path.join(input_dir, latest_file)
        data = pd.read_pickle(filepath)
        print(f"📊 Dataset shape: {data.shape}")
        print(f"📋 Columns: {data.columns.tolist()}")
        
        if len(data) == 0:
            print("❌ Empty dataset!")
            return
        
        # Phân tích chi tiết
        print(f"\n📈 DATASET OVERVIEW:")
        print(f"  - Total functions: {len(data)}")
        
        if 'target' in data.columns:
            target_dist = data['target'].value_counts().to_dict()
            print(f"  - Target distribution: {target_dist}")
            print(f"  - Balance ratio: {data['target'].mean():.3f}")
        
        # Phân tích input data (PyG Data objects)
        if 'input' in data.columns:
            print(f"\n🔍 GRAPH STRUCTURE ANALYSIS:")
            
            node_counts = []
            edge_counts = []
            feature_dims = []
            zero_nodes_counts = []
            edge_densities = []
            
            # Kiểm tra sample đầu tiên
            sample_input = data.iloc[0]['input']
            print(f"  - Sample data type: {type(sample_input)}")
            
            if hasattr(sample_input, 'x') and hasattr(sample_input, 'edge_index'):
                print(f"  - Sample x shape: {sample_input.x.shape}")
                print(f"  - Sample edge_index shape: {sample_input.edge_index.shape}")
                print(f"  - Sample y: {sample_input.y}")
                
                # Kiểm tra node features
                sample_x = sample_input.x
                if len(sample_x.shape) == 2:
                    print(f"  - Features per node: {sample_x.shape[1]}")
                    
                    # Kiểm tra node types (first feature)
                    node_types = sample_x[:, 0]
                    print(f"  - Node type range: [{node_types.min().item():.1f}, {node_types.max().item():.1f}]")
                    
                    # Kiểm tra zero nodes
                    zero_nodes = (sample_x == 0).all(dim=1).sum().item()
                    print(f"  - Zero nodes in sample: {zero_nodes}/{sample_x.shape[0]}")
            
            # Phân tích toàn bộ dataset
            print(f"\n📊 FULL DATASET STATISTICS:")
            
            for idx, row in data.iterrows():
                if idx >= 50:  # Chỉ check 50 samples đầu để tránh chậm
                    break
                    
                input_data = row['input']
                if not hasattr(input_data, 'x') or not hasattr(input_data, 'edge_index'):
                    continue
                
                num_nodes = input_data.x.shape[0]
                num_edges = input_data.edge_index.shape[1]
                feature_dim = input_data.x.shape[1] if len(input_data.x.shape) == 2 else 0
                
                node_counts.append(num_nodes)
                edge_counts.append(num_edges)
                feature_dims.append(feature_dim)
                
                # Đếm zero nodes
                if len(input_data.x.shape) == 2:
                    zero_nodes = (input_data.x == 0).all(dim=1).sum().item()
                    zero_nodes_counts.append(zero_nodes)
                
                # Tính edge density
                if num_nodes > 1:
                    max_edges = num_nodes * (num_nodes - 1)
                    edge_density = num_edges / max_edges if max_edges > 0 else 0
                    edge_densities.append(edge_density)
            
            if node_counts:
                print(f"  - Node count stats (n={len(node_counts)}):")
                print(f"    Min: {min(node_counts)}")
                print(f"    Mean: {np.mean(node_counts):.1f}")
                print(f"    Median: {np.median(node_counts):.1f}")
                print(f"    Max: {max(node_counts)}")
                print(f"    Std: {np.std(node_counts):.1f}")
            
            if edge_counts:
                print(f"  - Edge count stats:")
                print(f"    Min: {min(edge_counts)}")
                print(f"    Mean: {np.mean(edge_counts):.1f}")
                print(f"    Median: {np.median(edge_counts):.1f}")
                print(f"    Max: {max(edge_counts)}")
                print(f"    Std: {np.std(edge_counts):.1f}")
                
                # Kiểm tra có functions nào 0 edges không
                zero_edge_count = sum(1 for x in edge_counts if x == 0)
                print(f"    Functions with 0 edges: {zero_edge_count}/{len(edge_counts)}")
            
            if edge_densities:
                print(f"  - Edge density stats:")
                print(f"    Min: {min(edge_densities):.4f}")
                print(f"    Mean: {np.mean(edge_densities):.4f}")
                print(f"    Max: {max(edge_densities):.4f}")
            
            if zero_nodes_counts:
                total_zero_nodes = sum(zero_nodes_counts)
                total_nodes = sum(node_counts)
                print(f"  - Zero nodes: {total_zero_nodes}/{total_nodes} ({100*total_zero_nodes/total_nodes:.1f}%)")
            
            if feature_dims:
                unique_dims = set(feature_dims)
                print(f"  - Feature dimensions: {unique_dims}")
        
        # So sánh với expected values
        print(f"\n✅ VALIDATION:")
        
        if edge_counts:
            if min(edge_counts) > 0:
                print(f"  ✅ All functions have edges (min: {min(edge_counts)})")
            else:
                zero_edge_functions = sum(1 for x in edge_counts if x == 0)
                print(f"  ⚠️  {zero_edge_functions} functions still have 0 edges")
        
        if node_counts:
            if min(node_counts) > 0:
                print(f"  ✅ All functions have nodes (min: {min(node_counts)})")
            else:
                print(f"  ❌ Some functions have 0 nodes")
        
        if feature_dims and len(set(feature_dims)) == 1:
            print(f"  ✅ Consistent feature dimensions: {feature_dims[0]}")
        elif feature_dims:
            print(f"  ⚠️  Inconsistent feature dimensions: {set(feature_dims)}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if edge_counts and min(edge_counts) == 0:
            print(f"  - Some functions still have 0 edges. Check edge_type config.")
            print(f"  - Current config should be: 'edges.Ast,edges.Cfg'")
        
        if edge_counts and np.mean(edge_counts) < 10:
            print(f"  - Average edge count is low ({np.mean(edge_counts):.1f}). Consider adding more edge types.")
        
        if node_counts and (max(node_counts) / min(node_counts)) > 20:
            print(f"  - High node count variance ({min(node_counts)}-{max(node_counts)}). Variable size approach is correct.")
        
        if zero_nodes_counts and sum(zero_nodes_counts) > 0:
            print(f"  - Found zero nodes. This might indicate embedding issues.")
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        import traceback
        traceback.print_exc()

def compare_with_old_data():
    """So sánh với dữ liệu cũ"""
    print(f"\n🔄 COMPARING WITH OLD DATA:")
    
    input_dir = "data/input/"
    input_files = [f for f in os.listdir(input_dir) if f.endswith('_input.pkl')]
    
    if len(input_files) >= 2:
        # Sắp xếp theo thời gian
        input_files_with_time = []
        for f in input_files:
            filepath = os.path.join(input_dir, f)
            mtime = os.path.getmtime(filepath)
            input_files_with_time.append((f, mtime))
        
        input_files_with_time.sort(key=lambda x: x[1], reverse=True)
        
        if len(input_files_with_time) >= 2:
            latest_file = input_files_with_time[0][0]
            old_file = input_files_with_time[1][0]
            
            print(f"  📊 Comparing {latest_file} vs {old_file}")
            
            try:
                latest_data = pd.read_pickle(os.path.join(input_dir, latest_file))
                old_data = pd.read_pickle(os.path.join(input_dir, old_file))
                
                # So sánh edges
                latest_edges = [row['input'].edge_index.shape[1] for _, row in latest_data.iterrows()]
                old_edges = [row['input'].edge_index.shape[1] for _, row in old_data.iterrows()]
                
                print(f"  - Latest avg edges: {np.mean(latest_edges):.1f}")
                print(f"  - Old avg edges: {np.mean(old_edges):.1f}")
                print(f"  - Improvement: {np.mean(latest_edges) - np.mean(old_edges):.1f} edges")
                
                if np.mean(latest_edges) > np.mean(old_edges):
                    print(f"  ✅ Edge count improved!")
                else:
                    print(f"  ⚠️  Edge count not improved")
                    
            except Exception as e:
                print(f"  ❌ Error comparing: {e}")
    else:
        print(f"  📝 Only one input file found, skipping comparison")

if __name__ == "__main__":
    check_new_embedding_data()
    compare_with_old_data()
    print(f"\n🎯 SUMMARY: Check above for validation results and recommendations")
