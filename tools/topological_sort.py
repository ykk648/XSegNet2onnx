# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import sys
import onnx
'''
references:
https://zh.wikipedia.org/wiki/%E6%8B%93%E6%92%B2%E6%8E%92%E5%BA%8F
https://github.com/microsoft/onnxconverter-common/blob/ac76e26fad6ae69ad6d39e2acd9f6c2f6d14356d/onnxconverter_common/optimizer.py#L1509
'''


def _get_node_successor(node, name_nodes_map):
    successor = []
    for outp in node.output:
        if outp in name_nodes_map:
            successor.extend(name_nodes_map[outp])
    return successor


def _visit(node, name_nodes_map, node_status_map, res):
    if node_status_map[node.name] == 'perm':
        return
    if node_status_map[node.name] == 'temp':
        raise Exception("This graph is not a DAG")
    node_status_map[node.name] = 'temp'
    node_successor = _get_node_successor(node, name_nodes_map)
    for m in node_successor:
        _visit(m, name_nodes_map, node_status_map, res)
    node_status_map[node.name] = 'perm'
    res.insert(0, node)


def _topological_sort(nodes, name_nodes_map, node_status_map):
    res = []
    for node in nodes:
        _visit(node, name_nodes_map, node_status_map, res)
    return res


onnx_file = sys.argv[1]

ori_model = onnx.load(onnx_file)
ori_model = onnx.shape_inference.infer_shapes(ori_model)
graph = ori_model.graph
name_nodes_map = {}
node_status_map = {}
nodes = []

while len(graph.node):
    node = graph.node.pop()
    node_status_map.update({node.name: 'unmark'})
    for inp_name in node.input:
        name_nodes_map.setdefault(inp_name, []).append(node)
    nodes.append(node)

topo_nodes = _topological_sort(nodes, name_nodes_map, node_status_map)

graph.node.extend(topo_nodes)
onnx.save_model(ori_model, sys.argv[2])
