#import pygraphviz as pgv
import types
from pyvis.network import Network
import graphviz


def get_all_values(dict_, name_list):
    list_result = []
    for k, v in dict_.items():
        list_result = list(set(list_result) | set(dict_[k][name_list]))
    return list_result


def transform_text(text, text_transformations):
    for fun in text_transformations:
        text = fun(text)
    return text


class Node:

    def __init__(self, type, name, text_transformations=[], steps=[], show_steps=True):
        self.type = type
        self.name = name
        self.label = '\n'.join(steps) if show_steps else transform_text(name, text_transformations)

    def to_tuple(self):
        return (self.type, self.name, self.label)


def get_net_vars_from_dict(dict_, show_steps=True, text_transformations=[], endpoints_only=False):
    tasks = dict_.keys()
    outputs = get_all_values(dict_, "outputs_name")
    inputs = get_all_values(dict_, "inputs_name")

    if endpoints_only:
        enpoints_inputs = [i for i in inputs if i not in outputs]
        enpoints_outputs = [o for o in outputs if o not in inputs]

        outputs = enpoints_outputs
        inputs = enpoints_inputs

    nodes = []
    nodes += [Node('output', n, text_transformations=text_transformations, show_steps=False) for n in outputs]
    nodes += [Node('input', n, text_transformations=text_transformations, show_steps=False) for n in inputs]

    if endpoints_only:
        nodes.append(Node('task', 'pipeline', show_steps=False))
    else:
        nodes += [Node('task', n, steps=dict_[n]["steps"], show_steps=show_steps) for n in tasks]

    edges = []
    if endpoints_only:
        for nod in inputs:
            edges.append((nod, 'pipeline'))
        for nod in outputs:
            edges.append(('pipeline', nod))
    else:
        for k, v in dict_.items():
            for input_ in v["inputs_name"]:
                edges.append((input_, k))
            for output_ in v["outputs_name"]:
                edges.append((k, output_))

    return [n.to_tuple() for n in nodes], edges


def standard_add_node(net, **kwargs):
    return net.node(**kwargs)


def get_net_from_dict(dict_, *, net_type="agraph", show_steps=True,
                      rankdir="LR", text_transformations=[], endpoints_only=False):

    if net_type == "graphviz":
        net = graphviz.Digraph(format='svg', graph_attr={'rankdir': rankdir},
                               node_attr={'shape': 'rect'})

        # map graohviz interface to standard interface for add_node add_edge

        net.add_node = types.MethodType(standard_add_node, net)

        standard_add_edge = lambda self, source, target: self.edge(source, target)
        net.add_edge = types.MethodType(standard_add_edge, net)

    elif net_type == "agraph":
        raise ValueError("net_type 'agraph' not supported yet, change to 'pyvis' or 'graphviz'")
        # net = pgv.AGraph(directed=True, strict=True, rankdir=rankdir)
        # adjust a graph parameter
        # net.graph_attr["epsilon"] = "0.001"

    elif net_type == "pyvis":
        net = Network(notebook=True, directed=True)

    else:
        raise ValueError("net_type must be 'pyvis' or 'graphviz'")

    nodes, edges = get_net_vars_from_dict(dict_, show_steps,
                                          text_transformations=text_transformations, endpoints_only=endpoints_only)

    for type, name, label in nodes:
        shape_per_type = {
            'task': 'rect',
            'input': 'box',
            'output': 'box',
        }
        style_per_type = {
            'task': '',
            'input': 'rounded',
            'output': 'rounded',
        }
        color_per_type = {
            'task': '',
            'input': 'green',
            'output': 'red',
        }
        net.add_node(name=name, label=label, shape=shape_per_type[type], color=color_per_type[type], style=style_per_type[type])

    for source, target in edges:
        net.add_edge(source, target)

    return net

text_split_by_dot = lambda text: text.replace('.', '\n')

def display_graph(graph, show_steps=True, rankdir="LR", text_transformations=[text_split_by_dot], endpoints_only=False):
    return get_net_from_dict(graph, net_type="graphviz", show_steps=show_steps, rankdir=rankdir,
                             text_transformations=text_transformations, endpoints_only=endpoints_only)
