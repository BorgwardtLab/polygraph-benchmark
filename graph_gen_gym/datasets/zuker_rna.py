import networkx as nx
import seqfold
import torch
from torch_geometric.data import Data

BASE_TO_NUM = {"A": 0, "C": 1, "G": 2, "U": 3}
NUM_TO_BASE = {val: key for key, val in BASE_TO_NUM.items()}


def get_zuker_graph(rna=None, dotbracket=None):
    if dotbracket is None:
        assert rna is not None
        structs = seqfold.fold(rna)
        dotbracket = seqfold.dot_bracket(rna, structs)
    stack = []
    wattson_crick_index = []
    for idx, symbol in enumerate(dotbracket):
        if symbol == "(":
            stack.append(idx)
        elif symbol == ")":
            paired = stack.pop()
            wattson_crick_index.append([idx, paired])
        elif symbol == ".":
            continue
        else:
            raise ValueError
    wattson_crick_index = torch.Tensor(wattson_crick_index)
    wattson_crick_index = torch.cat(
        [wattson_crick_index, torch.flip(wattson_crick_index, dims=(1,))], dim=0
    ).T
    backbone_index = torch.stack(
        [torch.arange(0, len(dotbracket) - 1), torch.arange(1, len(dotbracket))], dim=1
    )
    backbone_index = torch.cat(
        [backbone_index, torch.flip(backbone_index, dims=(1,))], dim=0
    ).T
    edge_index = torch.cat([backbone_index, wattson_crick_index], dim=1).to(torch.int64)
    is_backbone = torch.cat(
        [torch.ones(backbone_index.size(1)), torch.zeros(wattson_crick_index.size(1))],
        dim=0,
    ).to(torch.int64)
    assert len(is_backbone) == edge_index.size(1)
    bases = torch.Tensor([BASE_TO_NUM[b] for b in rna]).to(torch.int64)
    return Data(
        edge_index=edge_index,
        bases=bases,
        is_backbone=is_backbone,
        num_nodes=len(dotbracket),
    )


def is_valid_pair(
    a: int, b: int, allow_wobble: bool = True, allow_all: bool = False
) -> bool:
    if allow_all:
        return True
    a, b = sorted([NUM_TO_BASE[a], NUM_TO_BASE[b]])
    canonical = (a == "A" and b == "U") or (a == "C" and b == "G")
    wobble = a == "G" and b == "U"
    if allow_wobble:
        return canonical or wobble
    return canonical


def is_valid(graph: nx.Graph, check_pairing: bool = True):
    backbone_graph = nx.Graph()
    backbone_graph.add_nodes_from(graph.nodes)
    backbone_graph.add_edges_from(
        e for e in graph.edges if graph.edges[e]["is_backbone"]
    )
    has_no_branches = max([d for n, d in backbone_graph.degree()]) <= 2
    is_path = nx.is_tree(backbone_graph) and has_no_branches
    if not is_path:
        return False
    endpoints = [n for n, d in backbone_graph.degree() if d == 1]
    assert len(endpoints) == 2
    backbone_path = nx.shortest_path(
        backbone_graph, source=endpoints[0], target=endpoints[1]
    )
    node_to_idx = {node: idx for idx, node in enumerate(backbone_path)}
    assert len(backbone_path) == graph.number_of_nodes(), (
        len(backbone_path),
        graph.number_of_nodes(),
    )

    # Now we check that there are no pseudoknots or weird strucutures
    pseudoknot_boundary_stack = []
    for idx, node in enumerate(backbone_path):
        neighbors = list(graph.neighbors(node))
        non_backbone_neighbors = [
            nb for nb in neighbors if not graph.edges[(node, nb)]["is_backbone"]
        ]
        if len(non_backbone_neighbors) > 1:
            return False
        if len(non_backbone_neighbors) > 0:
            assert len(non_backbone_neighbors) == 1
            nb = non_backbone_neighbors[0]
            nb_idx = node_to_idx[nb]

            if check_pairing and not is_valid_pair(
                graph.nodes[node]["bases"], graph.nodes[nb]["bases"]
            ):
                return False

            if nb_idx == idx:  # A self-loop
                return False
            if (
                len(pseudoknot_boundary_stack) > 0
                and nb_idx >= pseudoknot_boundary_stack[-1]
            ):
                return False

            if nb_idx > idx:
                pseudoknot_boundary_stack.append(nb_idx)
            else:
                assert (
                    len(pseudoknot_boundary_stack) > 0
                    and pseudoknot_boundary_stack[-1] == idx
                ), (pseudoknot_boundary_stack, idx, nb_idx)
                pseudoknot_boundary_stack.pop()

    assert len(pseudoknot_boundary_stack) == 0
    return True


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import networkx as nx
    from torch_geometric.utils import to_networkx

    result = to_networkx(
        get_zuker_graph(
            rna="GUCUACGGCCAUACCACCCUGAACGCGCCCGAUCUCGUCUGAUCUCGGAAGCUAAGCAGGGUCGGGCCUGGUUAGUACUUGGAUGGGAGACCGCCUGGGAAUACCGGGUGCUGUAGGCUUU",
            dotbracket="(((((((((....((((((((.....((((((............))))..))....)))))).)).(((((......((.((.(((....))))).)).....))))).)))))))))...",
        ),
        to_undirected=True,
        node_attrs=["bases"],
        edge_attrs=["is_backbone"],
    )
    # print([result.edges[e]["is_backbone"] for e in result.edges])
    print(is_valid(result))
    # nx.draw(result)
    # plt.savefig("testing.png")
    # print(result)
