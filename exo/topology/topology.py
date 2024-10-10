from .device_capabilities import DeviceCapabilities
from typing import Dict, List, Set, Optional, Tuple

class Topology:
  def __init__(self):
    self.nodes: Dict[str, DeviceCapabilities] = {}  # Maps node IDs to DeviceCapabilities
    self.peer_graph: Dict[str, Set[str]] = {}  # Adjacency list representing the graph
    self.active_node_id: Optional[str] = None
    self.latencies: Dict[Tuple[str, str], float] = {}
    self.ring: List[str] = []

  def update_node(self, node_id: str, device_capabilities: DeviceCapabilities):
    self.nodes[node_id] = device_capabilities
    if node_id not in self.peer_graph:
        self.peer_graph[node_id] = set()

  def get_node(self, node_id: str) -> Optional[DeviceCapabilities]:
    return self.nodes.get(node_id)

  def get_latency(self, node1_id: str, node2_id: str) -> float:
    return self.latencies.get((node1_id, node2_id), float('inf'))
    
  def get_average_latency(self, node_id: str) -> float:
    neighbors = self.get_neighbors(node_id)
    if not neighbors:
        return float('inf')
    return sum(self.get_latency(node_id, neighbor) for neighbor in neighbors) / len(neighbors)
        
  def all_nodes(self):
    return self.nodes.items()

  def add_edge(self, node1_id: str, node2_id: str, latency: float = 0.0):
    if node1_id not in self.peer_graph:
      self.peer_graph[node1_id] = set()
    if node2_id not in self.peer_graph:
      self.peer_graph[node2_id] = set()
    self.peer_graph[node1_id].add(node2_id)
    self.peer_graph[node2_id].add(node1_id)
    self.update_edge(node1_id, node2_id, latency)

  def update_edge(self, node1_id: str, node2_id: str, latency: float = 0.0):
    self.latencies[(node1_id, node2_id)] = self.latencies[(node2_id, node1_id)] = latency

  def get_neighbors(self, node_id: str) -> Set[str]:
    return self.peer_graph.get(node_id, set())

  def all_edges(self):
    edges = []
    for node, neighbors in self.peer_graph.items():
      for neighbor in neighbors:
        if (neighbor, node) not in edges:  # Avoid duplicate edges
          edges.append((node, neighbor))
    return edges

  def merge(self, other: "Topology"):
    for node_id, capabilities in other.nodes.items():
      self.update_node(node_id, capabilities)
    for node_id, neighbors in other.peer_graph.items():
      for neighbor in neighbors:
         latency = other.get_latency(node_id, neighbor)
         self.add_edge(node_id, neighbor, latency)

  def __str__(self):
    nodes_str = ", ".join(f"{node_id}: {cap}" for node_id, cap in self.nodes.items())
    edges_str = ", ".join(f"{node}: {neighbors}" for node, neighbors in self.peer_graph.items())
    ring_str = " -> ".join(self.ring) if self.ring else "Not set"
    latency_str = ", ".join(f"{(node1, node2)}: {latency}" for (node1, node2), latency in self.latencies.items())
    return f"Topology(Nodes: {{{nodes_str}}}, Edges: {{{edges_str}}}, Ring: {ring_str}, Latencies: {{{latency_str}}})"