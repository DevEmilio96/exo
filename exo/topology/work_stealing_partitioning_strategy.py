import asyncio
import math
from typing import List, Dict, Optional, Set
from exo.topology.partitioning_strategy import PartitioningStrategy, Partition
from exo.topology.topology import Topology
from exo.inference.shard import Shard
import logging

logger = logging.getLogger(__name__)

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
        node_capabilities = [self._calculate_node_capability(node_id) for node_id in nodes]
        total_capability = sum(node_capabilities)

        partitions = []
        cumulative = 0.0

        if total_capability <= 0:
            logger.warning("Total capability is zero or negative. Using equal partitions.")
            partition_size = 1.0 / len(nodes)
            for i, node_id in enumerate(nodes):
                partitions.append(Partition(node_id=node_id, start=i*partition_size, end=(i+1)*partition_size))
                self._initialize_node(node_id, partition_size)
        else:
            for node_id, capability in zip(nodes, node_capabilities):
                node_partition = capability / total_capability
                partitions.append(Partition(node_id=node_id, start=cumulative, end=cumulative + node_partition))
                self._initialize_node(node_id, node_partition)
                cumulative += node_partition

        print(partitions)
        return [Partition(node_id='d8c12d93-19d4-4cf8-83ad-10c42b50d320', start=0.0, end=0.99), Partition(node_id='385feab5-355f-4247-ba00-e3b41f99e7a8', start=0.99, end=1.0)]

    def _initialize_node(self, node_id: str, capability: float):
        self.work_queue[node_id] = []
        self.locks[node_id] = asyncio.Lock()
        self.node_capabilities[node_id] = capability
        self.active_nodes.add(node_id)

    def add_shards(self, shards: List[Shard]):
        if not self.active_nodes:
            raise Exception("No active nodes available to assign shards.")

        shard_work = [shard.end_layer - shard.start_layer + 1 for shard in shards]
        total_work = sum(shard_work)
        total_capability = sum(self.node_capabilities.values())

        if total_capability <= 0:
            logger.warning("Total capability is zero or negative. Distributing work equally.")
            work_per_node = len(shards) / len(self.active_nodes)
            for i, node_id in enumerate(self.active_nodes):
                start = int(i * work_per_node)
                end = int((i + 1) * work_per_node)
                self.work_queue[node_id].extend(shards[start:end])
        else:
            target_work_per_node = {
                node_id: (self.node_capabilities[node_id] / total_capability) * total_work
                for node_id in self.active_nodes
            }

            shards_with_work = list(zip(shards, shard_work))
            work_assigned = {node_id: 0 for node_id in self.active_nodes}

            for shard, work in shards_with_work:
                node_id = min(
                    self.active_nodes,
                    key=lambda nid: (
                        work_assigned[nid] / max(target_work_per_node[nid], 1e-10),
                        self.topology.get_average_latency(nid),
                        -self.topology.nodes[nid].memory / max(self.memory_for_layer, 1e-10)
                    )
                )
                self.work_queue[node_id].append(shard)
                work_assigned[node_id] += work

    async def get_work(self, node_id: str) -> Optional[Shard]:
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
                            logger.info(f"Node {node_id} stole work from {donor_node_id}")
                            logger.debug(f"Work distribution: {node_work_counts}")
                            return stolen_shard

            if self.work_queue.get(node_id):
                return self.work_queue[node_id].pop()

        return None

    async def add_work(self, node_id: str, shard: Shard):
        if node_id in self.active_nodes:
            async with self.locks[node_id]:
                self.work_queue[node_id].append(shard)

    def is_work_remaining(self) -> bool:
        return any(
            self.work_queue.get(nid) and self.topology.nodes[nid].is_active
            for nid in self.active_nodes
        )

    def _calculate_node_capability(self, node_id: str) -> float:
        node = self.topology.nodes[node_id]
        flops_factor = max(node.flops.fp16, 1e-10)
        
        # Nuovo calcolo del memory_factor
        available_layers = node.memory / max(self.memory_for_layer, 1e-10)
        memory_factor = 1 - math.exp(-available_layers / 10)  # Usiamo una funzione esponenziale per limitare la crescita
        
        capability = flops_factor * memory_factor
        logger.debug(f"Node {node_id} capability: {capability:.5f} (FLOPS: {flops_factor:.2f}, memory_factor: {memory_factor:.5f}, memory: {node.memory}, memory_for_layer: {self.memory_for_layer})")
        return capability