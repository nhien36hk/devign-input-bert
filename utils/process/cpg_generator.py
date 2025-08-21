import json
import re
import subprocess
import os.path
import os
import time
# from .cpg_client_wrapper import CPGClientWrapper
#from ..data import datamanager as data


# def funcs_to_graphs(funcs_path):
#     client = CPGClientWrapper()
#     # query the cpg for the dataset
#     print(f"Creating CPG.")
#     graphs_string = client(funcs_path)
#     # removes unnecessary namespace for object references
#     graphs_string = re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", '', graphs_string)
#     graphs_json = json.loads(graphs_string)

#     return graphs_json["functions"]


def graph_indexing(graph):
    idx = int(graph["file"].split(".c")[0].split("/")[-1])
    del graph["file"]
    return idx, {"functions": [graph]}


def joern_parse(joern_path, input_path, output_path, file_name):
    out_file = file_name + ".bin"
    joern_parse_call = subprocess.run(["./" + joern_path + "joern-parse", input_path, "--out", output_path + out_file],
                                      stdout=subprocess.PIPE, text=True, check=True)
    print(str(joern_parse_call))
    return out_file


def joern_create(joern_path, in_path, out_path, cpg_files):
    json_files = []
    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "graph-for-funcs.sc")
    
    # Chuẩn bị TẤT CẢ lệnh trước
    all_commands = []
    for cpg_file in cpg_files:
        json_file_name = f"{cpg_file.split('.')[0]}.json"
        json_files.append(json_file_name)
        
        if os.path.exists(in_path + cpg_file):
            json_out = f"{os.path.abspath(out_path)}/{json_file_name}"
            # Thêm 3 lệnh cho mỗi file
            all_commands.extend([
                f"importCpg(\"{os.path.abspath(in_path)}/{cpg_file}\")",
                f"cpg.runScript(\"{script_path}\").toString() |> \"{json_out}\"", 
                "delete"
            ])
    
    # Thêm lệnh exit
    all_commands.append("exit")

    input_text = "\n".join(all_commands) + "\n"
    
    try:
        #  Chạy process và đọc HẾT output cùng lúc  
        joern_process = subprocess.Popen(["./" + joern_path + "joern"], 
                                       stdin=subprocess.PIPE, 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE,
                                       text=True)
        
        # communicate() = gửi hết + đọc hết, NO DEADLOCK!
        outs, errs = joern_process.communicate(input=input_text, timeout=1800)  # 30 phút

        if errs:
            print(f"⚠️ Warnings: {errs}")
            
    except subprocess.TimeoutExpired:
        joern_process.kill()
        joern_process.communicate()
        print("❌ Timeout after 30 minutes")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return json_files


def json_process(in_path, json_file):
    if os.path.exists(in_path+json_file):
        with open(in_path+json_file) as jf:
            cpg_string = jf.read()
            cpg_string = re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", '', cpg_string)
            cpg_json = json.loads(cpg_string)
            container = [graph_indexing(graph) for graph in cpg_json["functions"] if graph["file"] != "N/A"]
            return container
    return None

'''
def generate(dataset, funcs_path):
    dataset_size = len(dataset)
    print("Size: ", dataset_size)
    graphs = funcs_to_graphs(funcs_path[2:])
    print(f"Processing CPG.")
    container = [graph_indexing(graph) for graph in graphs["functions"] if graph["file"] != "N/A"]
    graph_dataset = data.create_with_index(container, ["Index", "cpg"])
    print(f"Dataset processed.")

    return data.inner_join_by_index(dataset, graph_dataset)
'''

# client = CPGClientWrapper()
# client.create_cpg("../../data/joern/")
# joern_parse("../../joern/joern-cli/", "../../data/joern/", "../../joern/joern-cli/", "gen_test")
# print(funcs_to_graphs("/data/joern/"))
"""
while True:
    raw = input("query: ")
    response = client.query(raw)
    print(response)
"""
