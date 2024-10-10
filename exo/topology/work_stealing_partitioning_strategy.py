import asyncio
from typing import List, Dict, Optional, Set
from exo.topology.partitioning_strategy import PartitioningStrategy, Partition
from exo.topology.topology import Topology
from exo.inference.shard import Shard

class WorkStealingPartitioningStrategy(PartitioningStrategy):
    def __init__(self):
        self.work_queue: Dict[str, List[Shard]] = {}
        self.node_capabilities: Dict[str, float] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        self.active_nodes: Set[str] = set()
        self.topology: Optional[Topology] = None
        self.memory_for_layer: float = 500  # Default value

    def partition(self, topology: Topology) -> List[Partition]:
        self.topology = topology
        nodes = list(topology.nodes.keys())
        total_capability = sum(self._calculate_node_capability(node_id) for node_id in nodes)
        partitions = []
        cumulative = 0.0
        
        for node_id in nodes:
            capability = self._calculate_node_capability(node_id) / total_capability
            self.work_queue[node_id] = []
            self.locks[node_id] = asyncio.Lock()
            self.node_capabilities[node_id] = capability
            self.active_nodes.add(node_id)
            partitions.append(Partition(node_id=node_id, start=cumulative, end=cumulative + capability))
            cumulative += capability
        
        # Ensure all nodes in the topology have a partition
        for node_id in topology.nodes.keys():
            if node_id not in [p.node_id for p in partitions]:
                partitions.append(Partition(node_id=node_id, start=cumulative, end=cumulative))
                cumulative += 1e-10  # Add a small value to ensure unique end points
        
        return partitions

    def add_shards(self, shards: List[Shard]):
        if not self.active_nodes:
            raise Exception("No active nodes available to assign shards.")

        shard_work = [shard.end_layer - shard.start_layer + 1 for shard in shards]
        total_work = sum(shard_work)
        total_capability = sum(self.node_capabilities.values())

        target_work_per_node = {
            node_id: self.node_capabilities[node_id] / total_capability * total_work
            for node_id in self.active_nodes
        }

        shards_with_work = list(zip(shards, shard_work))
        work_assigned = {node_id: 0 for node_id in self.active_nodes}

        for shard, work in shards_with_work:
            node_id = min(
                self.active_nodes,
                key=lambda nid: (
                    work_assigned[nid] / target_work_per_node[nid] if target_work_per_node[nid] > 0 else float('inf'),
                    self.topology.get_average_latency(nid),
                    -self.topology.nodes[nid].memory / self.memory_for_layer
                )
            )
            self.work_queue[node_id].append(shard)
            work_assigned[node_id] += work

    async def get_work(self, node_id: str) -> Optional[Shard]:
        node_capabilities = self.topology.get_node(node_id)
        if not node_capabilities:
            print(f"Node {node_id} capabilities not found in topology.")
            return None

        print(f"Node {node_id} capabilities: Memory={node_capabilities.memory} MB")

        # Log the current work queue
        print(f"Current work queue has {len(self.work_queue)} shards.")

        # Iterate through the work queue to find a suitable shard
        for shard in self.work_queue:
            required_memory = self.estimate_memory_usage(shard)
            print(f"Evaluating shard {shard}: requires {required_memory} MB memory.")

            if required_memory <= node_capabilities.memory:
                print(f"Assigning shard {shard} to node {node_id}.")
                self.work_queue.remove(shard)
                return shard
            else:
                print(f"Shard {shard} requires more memory than node {node_id} has.")

        print(f"No suitable shard found for node {node_id} due to memory constraints.")
        if node_id not in self.active_nodes:
            print(f"Warning: Node {node_id} is not in active_nodes. Adding it.")
            self.active_nodes.add(node_id)
            self.work_queue[node_id] = []
            self.locks[node_id] = asyncio.Lock()
            self.node_capabilities[node_id] = self._calculate_node_capability(node_id)

        async with self.locks[node_id]:
            my_work_count = len(self.work_queue.get(node_id, []))
            if my_work_count <= 2:
                node_work_counts = {nid: len(self.work_queue.get(nid, [])) for nid in self.active_nodes}
                potential_donors = [
                    nid for nid in self.active_nodes
                    if nid != node_id and node_work_counts.get(nid, 0) > my_work_count
                ]

                potential_donors.sort(key=lambda nid: -node_work_counts.get(nid, 0))

                for donor_node_id in potential_donors:
                    async with self.locks[donor_node_id]:
                        donor_work_count = len(self.work_queue.get(donor_node_id, []))
                        if donor_work_count > my_work_count:
                            stolen_shard = self.work_queue[donor_node_id].pop(0)
                            print(f"Node {node_id} stole work from {donor_node_id}")
                            print(f"Work distribution: {node_work_counts}")
                            return stolen_shard

            if self.work_queue.get(node_id):
                return self.work_queue[node_id].pop()
        return None

    async def add_work(self, node_id: str, shard: Shard):
        if node_id not in self.active_nodes:
            print(f"Warning: Adding work to inactive node {node_id}. Activating it.")
            self.active_nodes.add(node_id)
            self.work_queue[node_id] = []
            self.locks[node_id] = asyncio.Lock()
            self.node_capabilities[node_id] = self._calculate_node_capability(node_id)

        async with self.locks[node_id]:
            self.work_queue[node_id].append(shard)

    def is_work_remaining(self) -> bool:
        return any(
            self.work_queue.get(nid) and self.topology.nodes[nid].is_active
            for nid in self.active_nodes
        )

    def _calculate_node_capability(self, node_id: str) -> float:
        node = self.topology.nodes[node_id]
        flops_factor = node.flops.fp16
        memory_factor = node.memory / self.memory_for_layer
        return flops_factor * memory_factor

    def ensure_node_initialized(self, node_id: str):
        if node_id not in self.active_nodes:
            print(f"Initializing node {node_id}")
            self.active_nodes.add(node_id)
            self.work_queue[node_id] = []
            self.locks[node_id] = asyncio.Lock()
            self.node_capabilities[node_id] = self._calculate_node_capability(node_id)